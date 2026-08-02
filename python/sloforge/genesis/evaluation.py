"""Artifact-derived multi-seed Genesis CPU evaluation with explicit non-results."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import statistics
import sys
import time
from collections.abc import Sequence
from datetime import timedelta
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from sloforge.genesis.capsule import (
    Digest,
    GenesisCapsule,
    ValidationContext,
    load_capsule,
    seal_capsule,
    validate_capsule,
)
from sloforge.genesis.ir import canonical_json

from .demo import GenesisDemoResult, run_genesis_demo


class HypothesisStatus(StrEnum):
    SUPPORTED_IN_SCOPE = "supported_in_scope"
    PARTIALLY_EVALUATED = "partially_evaluated"
    NOT_EVALUATED = "not_evaluated"


class HypothesisReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    hypothesis_id: str
    statement: str
    status: HypothesisStatus
    scope: str
    metrics: dict[str, float | int | str | bool]
    evidence_paths: tuple[str, ...]
    limitations: tuple[str, ...]


class EvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: str = "1.0.0"
    seed: int
    seeds: tuple[int, ...]
    run_count: int
    accepted_candidate_runtime_differential_rate: float
    accepted_runtime_success_rate: float
    real_rejection_rate: float
    capsule_acceptance_rate: float
    external_production_eligibility_rate: float
    redteam_replay_rate: float
    evolution_promotion_rate: float
    kernel_speedup_claims: int
    hardware_backed_runs: int
    hypotheses: tuple[HypothesisReport, ...]
    report_path: str


class EvaluationRates(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    accepted_runtime_success_rate: float
    local_capsule_acceptance_rate: float
    external_production_eligibility_rate: float
    local_evolution_promotion_rate: float


class _RateRun(Protocol):
    accepted_candidate_id: str
    accepted_genome_hash: str
    runtime_differential_passed: bool
    capsule_local_evolution_eligible: bool
    capsule_external_production_eligible: bool
    evolution_promoted: bool


def derive_evaluation_rates(runs: Sequence[_RateRun]) -> EvaluationRates:
    """Derive explicitly scoped local/external rates without promotion substitution."""

    if not runs:
        raise ValueError("evaluation rates require at least one run")
    accepted = tuple(
        bool(item.accepted_candidate_id)
        and bool(item.accepted_genome_hash)
        and item.runtime_differential_passed
        for item in runs
    )
    count = len(runs)
    return EvaluationRates(
        accepted_runtime_success_rate=sum(accepted) / count,
        local_capsule_acceptance_rate=sum(
            accepted[index] and item.capsule_local_evolution_eligible
            for index, item in enumerate(runs)
        )
        / count,
        external_production_eligibility_rate=sum(
            accepted[index] and item.capsule_external_production_eligible
            for index, item in enumerate(runs)
        )
        / count,
        local_evolution_promotion_rate=sum(
            accepted[index] and item.capsule_local_evolution_eligible and item.evolution_promoted
            for index, item in enumerate(runs)
        )
        / count,
    )


def _safe_reset(path: Path, repository: Path) -> None:
    if not path.exists():
        return
    if path.is_symlink():
        raise ValueError(f"refusing to reset symlinked evaluation path: {path}")
    resolved = path.resolve()
    if (
        resolved in {Path("/").resolve(), Path.home().resolve(), repository.resolve()}
        or len(resolved.parts) < 4
    ):
        raise ValueError(f"refusing to reset unsafe evaluation path: {resolved}")
    shutil.rmtree(resolved)


def _write(path: Path, value: BaseModel | dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite evaluation artifact: {path}")
    path.write_bytes(canonical_json(value) + b"\n")


def _tamper_gate(run: GenesisDemoResult, output: Path) -> dict[str, object]:
    capsule_path = Path(run.capsule_path)
    root = capsule_path.parent.parent
    context_path = root.with_name(f"{root.name}.validation-context.json")
    capsule = load_capsule(capsule_path)
    context = ValidationContext.model_validate_json(context_path.read_bytes(), strict=True)
    tests: dict[str, list[str]] = {}

    def codes(document: GenesisCapsule, current: ValidationContext) -> list[str]:
        report = validate_capsule(document, root, current)
        return sorted({item.code.value for item in report.issues})

    tests["modified_manifest"] = codes(
        capsule.model_copy(update={"known_unsupported_cases": ("tampered",)}), context
    )
    tests["mismatched_hardware"] = codes(
        capsule,
        context.model_copy(update={"hardware_fingerprint": Digest(value="0" * 64)}),
    )
    tests["stale_evidence"] = codes(
        capsule,
        context.model_copy(update={"now": context.now + timedelta(days=60)}),
    )
    incomplete = seal_capsule(
        capsule.model_copy(
            update={
                "capsule_digest": None,
                "evidence": tuple(
                    item for item in capsule.evidence if item.evidence_class.value != "quality"
                ),
            }
        )
    )
    tests["incomplete_evidence"] = codes(incomplete, context)
    result = {
        "schema_version": "1.0.0",
        "candidate_id": run.accepted_candidate_id,
        "tests": tests,
        "all_rejected": all(bool(value) for value in tests.values()),
        "raw_capsule_unchanged": True,
    }
    _write(output, result)
    return result


def run_genesis_evaluation(
    output: Path,
    *,
    seed: int,
    count: int,
    reset: bool = False,
) -> EvaluationResult:
    if seed < 0 or count < 2:
        raise ValueError("evaluation requires a non-negative seed and at least two runs")
    if output.exists() and output.is_symlink():
        raise ValueError(f"refusing to use symlinked evaluation path: {output}")
    repository = Path(__file__).resolve().parents[3]
    if reset:
        _safe_reset(output, repository)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"evaluation output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    seeds = tuple(seed + index for index in range(count))
    runs: list[GenesisDemoResult] = []
    elapsed_seconds: list[float] = []
    tamper_results: list[dict[str, object]] = []
    timing_paths: list[str] = []
    adapter_audit_paths: list[str] = []
    adapter_line_counts: list[int] = []
    for run_seed in seeds:
        started = time.monotonic()
        # Keep the reference model initialization fixed so that the held-out
        # expected outputs remain a valid semantic oracle while search,
        # adversarial, and benchmark schedules vary across seeds.
        run = run_genesis_demo(
            output / "runs" / str(run_seed),
            seed=run_seed,
            runtime_seed=seed,
        )
        elapsed = time.monotonic() - started
        elapsed_seconds.append(elapsed)
        timing_path = output / "timings" / f"seed-{run_seed}.json"
        _write(
            timing_path,
            {
                "schema_version": "1.0.0",
                "seed": run_seed,
                "elapsed_seconds": elapsed,
                "timer": "time.monotonic",
                "scope": "full local Genesis demo orchestration",
                "python": sys.version,
                "platform": platform.platform(),
                "hardware_backed": False,
            },
        )
        timing_paths.append(str(timing_path.resolve()))
        package_record = json.loads(
            (Path(run.output_directory) / "inspection/reference_package.json").read_text(
                encoding="utf-8"
            )
        )
        package_root = Path(str(package_record["package_root"]))
        adapter_sources = tuple(
            path
            for path in sorted(package_root.rglob("*.py"))
            if any(marker in path.name.lower() for marker in ("adapter", "runtime", "server"))
        )
        adapter_lines = sum(
            len(path.read_text(encoding="utf-8").splitlines()) for path in adapter_sources
        )
        adapter_audit_path = output / "adapter-audit" / f"seed-{run_seed}.json"
        _write(
            adapter_audit_path,
            {
                "schema_version": "1.0.0",
                "seed": run_seed,
                "package_root": str(package_root.resolve()),
                "package_hash": package_record["package_hash"],
                "model_specific_serving_sources": [
                    str(path.relative_to(package_root)) for path in adapter_sources
                ],
                "human_authored_model_specific_serving_lines": adapter_lines,
                "classification": "filename-based adapter/runtime/server source audit",
            },
        )
        adapter_audit_paths.append(str(adapter_audit_path.resolve()))
        adapter_line_counts.append(adapter_lines)
        runs.append(run)
        tamper_results.append(_tamper_gate(run, output / "tamper" / f"seed-{run_seed}.json"))
    run_paths = tuple(item.report_path for item in runs)
    candidate_runtime_differential_rate = (
        sum(item.runtime_differential_passed for item in runs) / count
    )
    rates = derive_evaluation_rates(runs)
    accepted_success = rates.accepted_runtime_success_rate
    capsule_acceptance = rates.local_capsule_acceptance_rate
    external_eligibility = rates.external_production_eligibility_rate
    rejections = sum(bool(item.rejected_candidate_ids) for item in runs) / count
    replay_count = sum(item.redteam_replayed_count for item in runs)
    finding_count = sum(item.redteam_finding_count for item in runs)
    promotion_rate = rates.local_evolution_promotion_rate
    hypothesis_directory = output / "hypotheses"

    reports = (
        HypothesisReport(
            hypothesis_id="H1",
            statement="Genesis synthesizes correct baseline servers for unseen generated tasks.",
            status=HypothesisStatus.PARTIALLY_EVALUATED,
            scope="one fixed HybridDecoder CPU package across search and schedule seeds",
            metrics={
                "run_count": count,
                "accepted_candidate_runtime_differential_rate": (
                    candidate_runtime_differential_rate
                ),
                "median_end_to_end_run_seconds": statistics.median(elapsed_seconds),
                "human_authored_production_adapter_lines": sum(adapter_line_counts),
            },
            evidence_paths=(*run_paths, *timing_paths, *adapter_audit_paths),
            limitations=(
                "this is not a randomized unseen-task campaign; one model package and one "
                "reference-model seed are reused",
                "the differential result is for the accepted candidate runtime assembled from "
                "the generated baseline, not a separate untouched-baseline rate",
                "PyTorch export and GPU paths were not exercised",
            ),
        ),
        HypothesisReport(
            hypothesis_id="H2",
            statement="Whole-stack synthesis outperforms configuration-only optimization.",
            status=HypothesisStatus.NOT_EVALUATED,
            scope="no hardware-comparable whole-stack performance campaign",
            metrics={"hardware_backed_runs": 0},
            evidence_paths=run_paths,
            limitations=(
                "the local capsule accepts no performance improvement claim and cannot establish H2",
            ),
        ),
        HypothesisReport(
            hypothesis_id="H3",
            statement="Counterexample-guided synthesis reduces escaped failures.",
            status=HypothesisStatus.PARTIALLY_EVALUATED,
            scope="deadline batching cancellation family",
            metrics={
                "runs_with_real_rejection_rate": rejections,
                "learned_constraint_reuse_rate": sum(
                    bool(item.learned_constraint_ids) for item in runs
                )
                / count,
            },
            evidence_paths=run_paths,
            limitations=("tests-only/fuzz-only/model-check-only ablations were not run",),
        ),
        HypothesisReport(
            hypothesis_id="H4",
            statement="Autopsy-guided mutation improves search efficiency.",
            status=HypothesisStatus.NOT_EVALUATED,
            scope="mapping and mutation guards are implemented, not campaign-tested here",
            metrics={"hardware_experiments": 0},
            evidence_paths=(),
            limitations=("guided/random/unrestricted multi-seed comparison is absent",),
        ),
        HypothesisReport(
            hypothesis_id="H5",
            statement="Lineage transfer reduces optimization cost on related tasks.",
            status=HypothesisStatus.NOT_EVALUATED,
            scope="SQLite transfer and invalidation mechanisms only",
            metrics={"transfer_campaigns": 0},
            evidence_paths=(),
            limitations=("empty/related/unrelated/stale lineage campaign is absent",),
        ),
        HypothesisReport(
            hypothesis_id="H6",
            statement="Proof-carrying promotion catches unsafe or incompatible artifacts.",
            status=(
                HypothesisStatus.SUPPORTED_IN_SCOPE
                if all(result["all_rejected"] is True for result in tamper_results)
                else HypothesisStatus.PARTIALLY_EVALUATED
            ),
            scope="manifest tamper, hardware mismatch, stale and incomplete evidence",
            metrics={
                "tamper_campaigns": count,
                "local_capsule_acceptance_rate": capsule_acceptance,
                "external_production_eligibility_rate": external_eligibility,
                "all_mutations_rejected": all(
                    result["all_rejected"] is True for result in tamper_results
                ),
            },
            evidence_paths=tuple(
                str((output / "tamper" / f"seed-{item}.json").resolve()) for item in seeds
            ),
            limitations=(
                "dependency and binary mutations are covered by focused tests, not this run",
            ),
        ),
        HypothesisReport(
            hypothesis_id="H7",
            statement="Continuous evolution adapts better than static deployment.",
            status=HypothesisStatus.PARTIALLY_EVALUATED,
            scope="deterministic local controller fixture",
            metrics={
                "local_fixture_promotion_rate": promotion_rate,
                "active_stream_preservation_rate": sum(
                    item.active_stream_preserved for item in runs
                )
                / count,
            },
            evidence_paths=run_paths,
            limitations=("no static/threshold/physical-only comparative traffic campaign",),
        ),
        HypothesisReport(
            hypothesis_id="H8",
            statement="Executable red team finds violations beyond normal tests.",
            status=HypothesisStatus.PARTIALLY_EVALUATED,
            scope="unsafe local fixture over five adversarial surfaces",
            metrics={
                "unique_findings": finding_count,
                "regression_replay_rate": replay_count / finding_count if finding_count else 0.0,
            },
            evidence_paths=run_paths,
            limitations=(
                "normal-test-only discovery baseline and time-to-violation are not compared",
            ),
        ),
        HypothesisReport(
            hypothesis_id="H9",
            statement="Cross-layer candidates outperform isolated optimization.",
            status=HypothesisStatus.NOT_EVALUATED,
            scope="cross-layer candidate structure and correctness; no performance comparison",
            metrics={
                "cross_layer_acceptance_rate": sum(item.cross_layer_accepted for item in runs)
                / count,
                "kernel_speedup_claims": sum(item.kernel_speedup_claim_count for item in runs),
            },
            evidence_paths=run_paths,
            limitations=(
                "no measured best-single-layer versus cross-layer performance comparison",
            ),
        ),
    )
    for report in reports:
        _write(hypothesis_directory / f"{report.hypothesis_id}.json", report)
    report_path = output / "evaluation.json"
    result = EvaluationResult(
        seed=seed,
        seeds=seeds,
        run_count=count,
        accepted_candidate_runtime_differential_rate=candidate_runtime_differential_rate,
        accepted_runtime_success_rate=accepted_success,
        real_rejection_rate=rejections,
        capsule_acceptance_rate=capsule_acceptance,
        external_production_eligibility_rate=external_eligibility,
        redteam_replay_rate=replay_count / finding_count if finding_count else 0.0,
        evolution_promotion_rate=promotion_rate,
        kernel_speedup_claims=sum(item.kernel_speedup_claim_count for item in runs),
        hardware_backed_runs=sum(item.hardware_backed for item in runs),
        hypotheses=reports,
        report_path=str(report_path.resolve()),
    )
    _write(report_path, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=73129)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--reset", action="store_true")
    arguments = parser.parse_args(argv)
    result = run_genesis_evaluation(
        arguments.output,
        seed=arguments.seed,
        count=arguments.count,
        reset=arguments.reset,
    )
    print(json.dumps(result.model_dump(mode="json"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EvaluationRates",
    "EvaluationResult",
    "HypothesisReport",
    "HypothesisStatus",
    "derive_evaluation_rates",
    "main",
    "run_genesis_evaluation",
]
