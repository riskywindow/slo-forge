"""Artifact-backed, multi-seed H8 executable red-team comparison.

The normal-suite arm executes one ordinary case on each existing red-team
surface.  The adversarial arm uses the production bounded generators, runner,
benchmark-integrity auditor, minimizer, counterexample conversion, and
regression replay path.  All time-to-discovery values are deterministic case
evaluation units; this module intentionally records no wall-clock or hardware
performance claim.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from sloforge.genesis.ir import canonical_hash, canonical_json
from sloforge.redteam import (
    AdversarialCase,
    BenchmarkAuditCase,
    RedTeamConfiguration,
    RedTeamReport,
    RedTeamSurface,
    RegressionCorpus,
    RegressionReplayResult,
    ResourceAdversarialCase,
    ResourceAdversaryConfiguration,
    ScheduleAdversarialCase,
    ScheduleAdversaryConfiguration,
    TensorAdversarialCase,
    TensorAdversaryConfiguration,
    TopologyAdversarialCase,
    TopologyAdversaryConfiguration,
    UnsafeStreamingCandidate,
    audit_benchmark_integrity,
    corpus_from_report,
    generate_resource_cases,
    generate_schedule_cases,
    generate_tensor_cases,
    generate_topology_cases,
    replay_regression_corpus,
    run_red_team,
    unsafe_benchmark_comparison,
)

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]

_TENSOR_CASES = 12
_SCHEDULE_CASES = 12
_TOPOLOGY_CASES = 6
_RESOURCE_CASES = 8
_MAXIMUM_FINDINGS = 32
_MAXIMUM_MINIMIZATION_EVALUATIONS = 128


class RedTeamCampaignValidationError(ValueError):
    """H8 campaign evidence is missing, changed, or does not replay."""


class _Model(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        allow_inf_nan=False,
    )


class CampaignScope(_Model):
    evaluator: Literal["unsafe_streaming_candidate_fixture_v1"] = (
        "unsafe_streaming_candidate_fixture_v1"
    )
    evidence_scope: Literal["deterministic_cpu_executable_fixture"] = (
        "deterministic_cpu_executable_fixture"
    )
    normal_suite: Literal["five_case_nominal_contract_suite"] = "five_case_nominal_contract_suite"
    adversarial_suite: Literal["bounded_existing_redteam_surfaces"] = (
        "bounded_existing_redteam_surfaces"
    )
    evaluation_unit: Literal["executed_case"] = "executed_case"
    hardware_backed_runs: Literal[0] = 0
    gpu_hours: Literal[0] = 0
    wall_clock_performance_recorded: Literal[False] = False


class NormalTestObservation(_Model):
    test_name: NonEmpty
    surface: RedTeamSurface
    case_id: NonEmpty
    case_hash: Sha256
    violated_contract: NonEmpty | None


class MinimizedFinding(_Model):
    finding_id: NonEmpty
    surface: RedTeamSurface
    violated_contract: NonEmpty
    counterexample_id: NonEmpty
    learned_constraint_id: NonEmpty
    original_size_units: PositiveInt
    minimized_size_units: PositiveInt
    minimization_evaluations: NonNegativeInt
    reduced: bool

    @model_validator(mode="after")
    def consistent_reduction(self) -> Self:
        if self.minimized_size_units > self.original_size_units:
            raise ValueError("minimized finding is larger than its source")
        if self.reduced != (self.minimized_size_units < self.original_size_units):
            raise ValueError("minimized finding reduction flag is inconsistent")
        return self


class ReplayArtifact(_Model):
    schema_version: Literal["sloforge.genesis.h8-replay/v1"] = "sloforge.genesis.h8-replay/v1"
    seed: NonNegativeInt
    results: tuple[RegressionReplayResult, ...]


class SeedResult(_Model):
    seed: NonNegativeInt
    configuration: RedTeamConfiguration
    normal_observations: tuple[NormalTestObservation, ...]
    normal_evaluation_steps: Literal[5]
    normal_violation_contracts: tuple[NonEmpty, ...]
    redteam_report_path: NonEmpty
    redteam_report_sha256: Sha256
    regression_corpus_path: NonEmpty
    regression_corpus_sha256: Sha256
    regression_corpus_digest: Sha256
    replay_path: NonEmpty
    replay_sha256: Sha256
    redteam_evaluation_steps: NonNegativeInt
    time_to_first_violation_evaluation_units: PositiveInt | None
    redteam_violation_contracts: tuple[NonEmpty, ...]
    unique_violation_family_count: NonNegativeInt
    findings: tuple[MinimizedFinding, ...]
    converted_regression_count: NonNegativeInt
    reproduced_regression_count: NonNegativeInt
    timed_out: bool

    @model_validator(mode="after")
    def internally_consistent(self) -> Self:
        if len(self.normal_observations) != self.normal_evaluation_steps:
            raise ValueError("normal observation count differs from evaluation steps")
        if tuple(sorted(set(self.redteam_violation_contracts))) != self.redteam_violation_contracts:
            raise ValueError("red-team contracts must be sorted and unique")
        if self.unique_violation_family_count != len(self.redteam_violation_contracts):
            raise ValueError("unique red-team family count is inconsistent")
        if self.converted_regression_count != len(self.findings):
            raise ValueError("every finding must be converted into a regression")
        if self.reproduced_regression_count > self.converted_regression_count:
            raise ValueError("reproduced regression count exceeds converted regressions")
        return self


class ContractRecurrence(_Model):
    violated_contract: NonEmpty
    detected_seeds: tuple[NonNegativeInt, ...]
    first_seed: NonNegativeInt
    later_recurrence_count: NonNegativeInt
    regression_reproduction_count: NonNegativeInt

    @model_validator(mode="after")
    def valid_recurrence(self) -> Self:
        if not self.detected_seeds or self.first_seed != self.detected_seeds[0]:
            raise ValueError("recurrence record has an invalid first seed")
        if self.later_recurrence_count != len(self.detected_seeds) - 1:
            raise ValueError("later recurrence count is inconsistent")
        return self


class _CampaignAggregates(_Model):
    normal_unique_violation_contracts: tuple[NonEmpty, ...]
    redteam_unique_violation_contracts: tuple[NonEmpty, ...]
    redteam_only_violation_contracts: tuple[NonEmpty, ...]
    redteam_unique_violation_family_count: NonNegativeInt
    normal_evaluation_steps: NonNegativeInt
    redteam_evaluation_steps: NonNegativeInt
    converted_regressions: NonNegativeInt
    reproduced_regressions: NonNegativeInt
    recurrence: tuple[ContractRecurrence, ...]
    conclusion: Literal["supported_in_declared_fixture_scope", "not_supported"]


class RedTeamCampaignReport(_Model):
    schema_version: Literal["sloforge.genesis.h8-campaign/v1"] = "sloforge.genesis.h8-campaign/v1"
    hypothesis_id: Literal["H8"] = "H8"
    statement: Literal[
        "The executable red team discovers violations not found by normal tests."
    ] = "The executable red team discovers violations not found by normal tests."
    seeds: tuple[NonNegativeInt, ...]
    scope: CampaignScope
    raw_results_path: NonEmpty
    raw_results_sha256: Sha256
    results: tuple[SeedResult, ...]
    normal_unique_violation_contracts: tuple[NonEmpty, ...]
    redteam_unique_violation_contracts: tuple[NonEmpty, ...]
    redteam_only_violation_contracts: tuple[NonEmpty, ...]
    redteam_unique_violation_family_count: NonNegativeInt
    normal_evaluation_steps: NonNegativeInt
    redteam_evaluation_steps: NonNegativeInt
    converted_regressions: NonNegativeInt
    reproduced_regressions: NonNegativeInt
    recurrence: tuple[ContractRecurrence, ...]
    conclusion: Literal["supported_in_declared_fixture_scope", "not_supported"]
    limitations: tuple[NonEmpty, ...]

    @model_validator(mode="after")
    def complete_campaign(self) -> Self:
        if len(self.seeds) < 3 or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("H8 campaign requires at least three unique seeds")
        if tuple(item.seed for item in self.results) != self.seeds:
            raise ValueError("H8 result order differs from declared seeds")
        if self.redteam_unique_violation_family_count != len(
            self.redteam_unique_violation_contracts
        ):
            raise ValueError("campaign violation family count is inconsistent")
        return self


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite H8 campaign artifact: {path}")
    path.write_bytes(payload)


def _configuration(seed: int) -> RedTeamConfiguration:
    return RedTeamConfiguration(
        seed=seed,
        maximum_findings=_MAXIMUM_FINDINGS,
        maximum_minimization_evaluations=_MAXIMUM_MINIMIZATION_EVALUATIONS,
        minimization_timeout_seconds=2.0,
        run_timeout_seconds=10.0,
        tensor_cases=_TENSOR_CASES,
        schedule_cases=_SCHEDULE_CASES,
        topology_cases=_TOPOLOGY_CASES,
        resource_cases=_RESOURCE_CASES,
    )


def _normal_suite(seed: int) -> tuple[NormalTestObservation, ...]:
    """Execute ordinary generated/derived cases through the real target APIs."""

    target = UnsafeStreamingCandidate()
    tensor = generate_tensor_cases(TensorAdversaryConfiguration(seed=seed + 101, maximum_cases=3))[
        2
    ]
    source_schedule = generate_schedule_cases(
        ScheduleAdversaryConfiguration(seed=seed + 102, maximum_cases=1)
    )[0]
    schedule = ScheduleAdversarialCase(
        case_id=f"{source_schedule.case_id}-normal-failure-prefix",
        events=source_schedule.events[:7],
    )
    topology = generate_topology_cases(
        TopologyAdversaryConfiguration(seed=seed + 103, maximum_cases=2)
    )[1]
    resource = generate_resource_cases(
        ResourceAdversaryConfiguration(
            seed=seed + 104,
            maximum_cases=1,
            maximum_device_bytes=target.descriptor.device_capacity_bytes,
            maximum_host_bytes=target.descriptor.host_capacity_bytes,
            maximum_queue_depth=target.descriptor.queue_capacity - 1,
            maximum_process_count=target.descriptor.process_limit,
        )
    )[0]

    evaluated = (
        ("contiguous_tensor_smoke", RedTeamSurface.TENSOR, tensor, target.evaluate_tensor(tensor)),
        (
            "failure_without_unsafe_retry",
            RedTeamSurface.PROTOCOL,
            schedule,
            target.evaluate_schedule(schedule),
        ),
        (
            "healthy_selected_route",
            RedTeamSurface.TOPOLOGY,
            topology,
            target.evaluate_topology(topology),
        ),
        (
            "within_declared_resource_bounds",
            RedTeamSurface.RESOURCE,
            resource,
            target.evaluate_resource(resource),
        ),
    )
    observations = [
        NormalTestObservation(
            test_name=name,
            surface=surface,
            case_id=case.case_id,
            case_hash=canonical_hash(case),
            violated_contract=(observation.violated_contract if observation is not None else None),
        )
        for name, surface, case, observation in evaluated
    ]
    unsafe_comparison = unsafe_benchmark_comparison()
    clean_candidate = unsafe_comparison.baseline.model_copy(
        update={"run_id": "normal-candidate-run", "candidate_id": target.descriptor.candidate_id}
    )
    clean_comparison = unsafe_comparison.model_copy(update={"candidate": clean_candidate})
    benchmark_issues = audit_benchmark_integrity(clean_comparison)
    observations.append(
        NormalTestObservation(
            test_name="matched_benchmark_manifest",
            surface=RedTeamSurface.BENCHMARK,
            case_id="normal-benchmark-comparison",
            case_hash=canonical_hash(clean_comparison),
            violated_contract=(
                f"benchmark.integrity.{benchmark_issues[0].code.value}"
                if benchmark_issues
                else None
            ),
        )
    )
    return tuple(observations)


def _case_size(case: AdversarialCase) -> int:
    if isinstance(case, ScheduleAdversarialCase):
        return len(case.events)
    if isinstance(case, TensorAdversarialCase):
        tensor = case.input
        elements = 1
        for dimension in tensor.shape:
            elements *= dimension
        return max(1, elements)
    if isinstance(case, TopologyAdversarialCase):
        topology = case.topology
        return max(1, topology.hosts * topology.devices_per_host)
    if isinstance(case, ResourceAdversarialCase):
        return 4
    if not isinstance(case, BenchmarkAuditCase):
        raise TypeError(f"unsupported H8 adversarial case: {type(case)!r}")
    return 1


def _surface_seed(seed: int, surface: RedTeamSurface) -> int:
    digest = hashlib.sha256(f"{seed}:{surface.value}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _first_violation_evaluation(
    report: RedTeamReport, configuration: RedTeamConfiguration
) -> int | None:
    positions: dict[str, int] = {}
    ordinal = 0
    generated = (
        generate_tensor_cases(
            TensorAdversaryConfiguration(
                seed=_surface_seed(configuration.seed, RedTeamSurface.TENSOR),
                maximum_cases=configuration.tensor_cases,
            )
        ),
        generate_schedule_cases(
            ScheduleAdversaryConfiguration(
                seed=_surface_seed(configuration.seed, RedTeamSurface.PROTOCOL),
                maximum_cases=configuration.schedule_cases,
            )
        ),
        generate_topology_cases(
            TopologyAdversaryConfiguration(
                seed=_surface_seed(configuration.seed, RedTeamSurface.TOPOLOGY),
                maximum_cases=configuration.topology_cases,
            )
        ),
        generate_resource_cases(
            ResourceAdversaryConfiguration(
                seed=_surface_seed(configuration.seed, RedTeamSurface.RESOURCE),
                maximum_cases=configuration.resource_cases,
            )
        ),
    )
    for cases in generated:
        for case in cases:
            ordinal += 1
            positions[case.case_id] = ordinal
    benchmark_ordinal = ordinal + 1
    discovered = tuple(
        benchmark_ordinal
        if finding.surface is RedTeamSurface.BENCHMARK
        else positions[finding.original_case.case_id]
        for finding in report.findings
    )
    return min(discovered) if discovered else None


def _finding_summary(report: RedTeamReport) -> tuple[MinimizedFinding, ...]:
    return tuple(
        MinimizedFinding(
            finding_id=finding.finding_id,
            surface=finding.surface,
            violated_contract=finding.observation.violated_contract,
            counterexample_id=finding.counterexample.counterexample_id,
            learned_constraint_id=finding.learned_constraint.constraint_id,
            original_size_units=_case_size(finding.original_case),
            minimized_size_units=_case_size(finding.minimized_case),
            minimization_evaluations=finding.minimization_evaluations,
            reduced=_case_size(finding.minimized_case) < _case_size(finding.original_case),
        )
        for finding in report.findings
    )


def _seed_result(
    *,
    seed: int,
    configuration: RedTeamConfiguration,
    normal: tuple[NormalTestObservation, ...],
    report: RedTeamReport,
    report_path: Path,
    corpus: RegressionCorpus,
    corpus_path: Path,
    replay: ReplayArtifact,
    replay_path: Path,
) -> SeedResult:
    normal_contracts = tuple(
        sorted({item.violated_contract for item in normal if item.violated_contract is not None})
    )
    contracts = tuple(
        sorted({finding.observation.violated_contract for finding in report.findings})
    )
    reproduced = sum(item.reproduced for item in replay.results)
    return SeedResult(
        seed=seed,
        configuration=configuration,
        normal_observations=normal,
        normal_evaluation_steps=5,
        normal_violation_contracts=normal_contracts,
        redteam_report_path=str(report_path.resolve()),
        redteam_report_sha256=_sha256(report_path),
        regression_corpus_path=str(corpus_path.resolve()),
        regression_corpus_sha256=_sha256(corpus_path),
        regression_corpus_digest=corpus.corpus_digest,
        replay_path=str(replay_path.resolve()),
        replay_sha256=_sha256(replay_path),
        redteam_evaluation_steps=report.evaluation_steps,
        time_to_first_violation_evaluation_units=_first_violation_evaluation(report, configuration),
        redteam_violation_contracts=contracts,
        unique_violation_family_count=len(contracts),
        findings=_finding_summary(report),
        converted_regression_count=len(corpus.cases),
        reproduced_regression_count=reproduced,
        timed_out=report.timed_out,
    )


def _recurrence(results: tuple[SeedResult, ...]) -> tuple[ContractRecurrence, ...]:
    contracts = sorted(
        {contract for result in results for contract in result.redteam_violation_contracts}
    )
    records: list[ContractRecurrence] = []
    for contract in contracts:
        detected = tuple(
            result.seed for result in results if contract in result.redteam_violation_contracts
        )
        replay_count = sum(
            1
            for result in results
            for finding in result.findings
            if finding.violated_contract == contract
            and result.reproduced_regression_count == result.converted_regression_count
        )
        records.append(
            ContractRecurrence(
                violated_contract=contract,
                detected_seeds=detected,
                first_seed=detected[0],
                later_recurrence_count=len(detected) - 1,
                regression_reproduction_count=replay_count,
            )
        )
    return tuple(records)


def _campaign_fields(results: tuple[SeedResult, ...]) -> _CampaignAggregates:
    normal = tuple(
        sorted({contract for result in results for contract in result.normal_violation_contracts})
    )
    redteam = tuple(
        sorted({contract for result in results for contract in result.redteam_violation_contracts})
    )
    redteam_only = tuple(contract for contract in redteam if contract not in set(normal))
    return _CampaignAggregates(
        normal_unique_violation_contracts=normal,
        redteam_unique_violation_contracts=redteam,
        redteam_only_violation_contracts=redteam_only,
        redteam_unique_violation_family_count=len(redteam),
        normal_evaluation_steps=sum(item.normal_evaluation_steps for item in results),
        redteam_evaluation_steps=sum(item.redteam_evaluation_steps for item in results),
        converted_regressions=sum(item.converted_regression_count for item in results),
        reproduced_regressions=sum(item.reproduced_regression_count for item in results),
        recurrence=_recurrence(results),
        conclusion=(
            "supported_in_declared_fixture_scope"
            if redteam_only and all(not item.timed_out for item in results)
            else "not_supported"
        ),
    )


def run_redteam_campaign(
    output_root: Path,
    *,
    seeds: tuple[int, ...] = (73129, 73130, 73131),
) -> RedTeamCampaignReport:
    """Execute and persist the bounded H8 comparison for at least three seeds."""

    if len(seeds) < 3 or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise ValueError("H8 campaign requires at least three unique non-negative seeds")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("H8 campaign output must be empty")
    output_root.mkdir(parents=True, exist_ok=True)
    target = UnsafeStreamingCandidate()
    comparison = unsafe_benchmark_comparison()
    results: list[SeedResult] = []
    for seed in seeds:
        configuration = _configuration(seed)
        normal = _normal_suite(seed)
        report = run_red_team(
            target=target,
            configuration=configuration,
            benchmark_comparison=comparison,
        )
        if report.timed_out:
            raise RuntimeError(f"red-team campaign seed {seed} exhausted its wall-clock guard")
        corpus = corpus_from_report(report)
        replay = ReplayArtifact(
            seed=seed,
            results=replay_regression_corpus(
                target=target,
                corpus=corpus,
                benchmark_comparison=comparison,
            ),
        )
        seed_root = output_root / "seeds" / str(seed)
        report_path = seed_root / "redteam-report.json"
        corpus_path = seed_root / "regression-corpus.json"
        replay_path = seed_root / "regression-replay.json"
        _write_once(report_path, canonical_json(report) + b"\n")
        _write_once(corpus_path, canonical_json(corpus) + b"\n")
        _write_once(replay_path, canonical_json(replay) + b"\n")
        results.append(
            _seed_result(
                seed=seed,
                configuration=configuration,
                normal=normal,
                report=report,
                report_path=report_path,
                corpus=corpus,
                corpus_path=corpus_path,
                replay=replay,
                replay_path=replay_path,
            )
        )

    result_tuple = tuple(results)
    raw_path = output_root / "raw" / "seed-results.jsonl"
    _write_once(raw_path, b"".join(canonical_json(item) + b"\n" for item in result_tuple))
    aggregates = _campaign_fields(result_tuple)
    campaign_report = RedTeamCampaignReport(
        seeds=seeds,
        scope=CampaignScope(),
        raw_results_path=str(raw_path.resolve()),
        raw_results_sha256=_sha256(raw_path),
        results=result_tuple,
        normal_unique_violation_contracts=aggregates.normal_unique_violation_contracts,
        redteam_unique_violation_contracts=aggregates.redteam_unique_violation_contracts,
        redteam_only_violation_contracts=aggregates.redteam_only_violation_contracts,
        redteam_unique_violation_family_count=aggregates.redteam_unique_violation_family_count,
        normal_evaluation_steps=aggregates.normal_evaluation_steps,
        redteam_evaluation_steps=aggregates.redteam_evaluation_steps,
        converted_regressions=aggregates.converted_regressions,
        reproduced_regressions=aggregates.reproduced_regressions,
        recurrence=aggregates.recurrence,
        conclusion=aggregates.conclusion,
        limitations=(
            "the target is an executable seeded unsafe fixture, not a population of production models",
            "the normal arm is a five-case nominal contract suite, not all repository tests",
            "time-to-discovery is reported in deterministic case evaluations, not wall-clock time",
            "all observations are CPU-local synthetic validation; no GPU or hardware result is claimed",
        ),
    )
    report_path = output_root / "report.json"
    _write_once(report_path, canonical_json(campaign_report) + b"\n")
    validate_redteam_campaign(campaign_report)
    return campaign_report


def validate_redteam_campaign(report: RedTeamCampaignReport | Path) -> None:
    """Reopen and independently replay every raw H8 campaign artifact."""

    try:
        value = (
            RedTeamCampaignReport.model_validate_json(report.read_bytes(), strict=True)
            if isinstance(report, Path)
            else report
        )
    except ValueError as error:
        raise RedTeamCampaignValidationError("H8 campaign report is invalid") from error
    raw_path = Path(value.raw_results_path)
    if not raw_path.is_file() or _sha256(raw_path) != value.raw_results_sha256:
        raise RedTeamCampaignValidationError("raw H8 seed result digest mismatch")
    try:
        reopened = tuple(
            SeedResult.model_validate_json(line, strict=True)
            for line in raw_path.read_text(encoding="utf-8").splitlines()
            if line
        )
    except ValueError as error:
        raise RedTeamCampaignValidationError("raw H8 seed results are invalid") from error
    if reopened != value.results:
        raise RedTeamCampaignValidationError("H8 report differs from raw seed results")

    target = UnsafeStreamingCandidate()
    comparison = unsafe_benchmark_comparison()
    reproduced_results: list[SeedResult] = []
    for result in reopened:
        if result.configuration != _configuration(result.seed):
            raise RedTeamCampaignValidationError("H8 red-team configuration changed")
        report_path = Path(result.redteam_report_path)
        corpus_path = Path(result.regression_corpus_path)
        replay_path = Path(result.replay_path)
        artifacts = (
            (report_path, result.redteam_report_sha256),
            (corpus_path, result.regression_corpus_sha256),
            (replay_path, result.replay_sha256),
        )
        if any(not path.is_file() or _sha256(path) != digest for path, digest in artifacts):
            raise RedTeamCampaignValidationError("H8 supporting artifact digest mismatch")
        try:
            stored_report = RedTeamReport.model_validate_json(report_path.read_bytes(), strict=True)
            stored_corpus = RegressionCorpus.model_validate_json(
                corpus_path.read_bytes(), strict=True
            )
            stored_replay = ReplayArtifact.model_validate_json(
                replay_path.read_bytes(), strict=True
            )
        except ValueError as error:
            raise RedTeamCampaignValidationError("H8 supporting artifact is invalid") from error
        normal = _normal_suite(result.seed)
        rerun_report = run_red_team(
            target=target,
            configuration=result.configuration,
            benchmark_comparison=comparison,
        )
        if rerun_report != stored_report:
            raise RedTeamCampaignValidationError(
                "red-team findings do not deterministically replay"
            )
        rebuilt_corpus = corpus_from_report(rerun_report)
        if rebuilt_corpus != stored_corpus:
            raise RedTeamCampaignValidationError("counterexample regression conversion changed")
        rerun_replay = ReplayArtifact(
            seed=result.seed,
            results=replay_regression_corpus(
                target=target,
                corpus=stored_corpus,
                benchmark_comparison=comparison,
            ),
        )
        if rerun_replay != stored_replay:
            raise RedTeamCampaignValidationError("regression counterexamples do not replay")
        reproduced_results.append(
            _seed_result(
                seed=result.seed,
                configuration=result.configuration,
                normal=normal,
                report=stored_report,
                report_path=report_path,
                corpus=stored_corpus,
                corpus_path=corpus_path,
                replay=stored_replay,
                replay_path=replay_path,
            )
        )
    reproduced = tuple(reproduced_results)
    if reproduced != reopened:
        raise RedTeamCampaignValidationError("H8 seed summaries do not recompute")
    expected = _campaign_fields(reproduced)
    for field in type(expected).model_fields:
        if getattr(value, field) != getattr(expected, field):
            raise RedTeamCampaignValidationError(f"H8 aggregate {field} does not recompute")


__all__ = [
    "CampaignScope",
    "ContractRecurrence",
    "MinimizedFinding",
    "NormalTestObservation",
    "RedTeamCampaignReport",
    "RedTeamCampaignValidationError",
    "ReplayArtifact",
    "SeedResult",
    "run_redteam_campaign",
    "validate_redteam_campaign",
]
