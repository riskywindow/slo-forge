"""Canonical artifact-backed orchestration for the Genesis evaluation campaigns.

The suite does not reinterpret raw evidence.  It runs each campaign through its
native runner, invokes its independent validator, snapshots every resulting
artifact in one content-addressed manifest, and derives scoped H1--H9 summaries
from the validated campaign reports.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from sloforge.genesis.capsule import (
    CapsuleValidationReport,
    GenesisCapsule,
    ValidationContext,
    load_capsule,
    validate_capsule,
)
from sloforge.genesis.demo import GenesisDemoResult
from sloforge.genesis.evaluation import (
    EvaluationResult,
    run_genesis_evaluation,
    validate_genesis_evaluation,
)
from sloforge.genesis.evaluation_campaigns.autopsy import (
    AutopsyCampaignReport,
    run_autopsy_guided_campaign,
    validate_autopsy_guided_campaign,
)
from sloforge.genesis.evaluation_campaigns.capsule_attacks import (
    run_capsule_attack_campaign,
    validate_capsule_attack_campaign,
)
from sloforge.genesis.evaluation_campaigns.cegis import (
    H3CampaignReport,
    VerificationStrategy,
    run_cegis_campaign,
    validate_cegis_campaign,
)
from sloforge.genesis.evaluation_campaigns.lineage import (
    H5LineageCampaignReport,
    run_h5_lineage_campaign,
    validate_h5_lineage_campaign,
)
from sloforge.genesis.evaluation_campaigns.redteam import (
    RedTeamCampaignReport,
    run_redteam_campaign,
    validate_redteam_campaign,
)
from sloforge.genesis.evaluation_campaigns.unseen import (
    H1CampaignConfiguration,
    H1CampaignReport,
    run_h1_unseen_campaign,
    validate_h1_unseen_campaign,
)
from sloforge.genesis.evaluation_campaigns.whole_stack import (
    WholeStackCampaignReport,
    run_whole_stack_campaign,
    validate_whole_stack_campaign,
)
from sloforge.genesis.evolution import CapsuleReference
from sloforge.genesis.ir import canonical_json
from sloforge.synthbench import GrammarConfiguration

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
MetricValue = int | float | str | bool

_MAXIMUM_CAMPAIGN_SEEDS = 32
_REPORT_NAME = "GENESIS_EVALUATION_SUITE.json"
_MANIFEST_NAME = "artifact-manifest.json"
_CONFIGURATION_NAME = "configuration.json"


class EvaluationSuiteValidationError(ValueError):
    """The suite manifest, a native campaign, or a derived claim is invalid."""


class EvaluationStatus(StrEnum):
    COMPLETED = "completed"
    PARTIALLY_EVALUATED = "partially_evaluated"
    NOT_EVALUATED = "not_evaluated"


class HypothesisOutcome(StrEnum):
    SUPPORTED_IN_DECLARED_SCOPE = "supported_in_declared_scope"
    NOT_SUPPORTED_IN_DECLARED_SCOPE = "not_supported_in_declared_scope"
    MIXED_IN_DECLARED_SCOPE = "mixed_in_declared_scope"
    DESCRIPTIVE_ONLY = "descriptive_only"
    UNAVAILABLE = "unavailable"


class _SuiteModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        allow_inf_nan=False,
    )


class EvaluationSuiteConfiguration(_SuiteModel):
    schema_version: Literal["sloforge.genesis.evaluation-suite-config/v1"] = (
        "sloforge.genesis.evaluation-suite-config/v1"
    )
    seed: NonNegativeInt = 73129
    core_run_count: Annotated[int, Field(ge=2, le=16)] = 2
    campaign_seed_count: Annotated[int, Field(ge=3, le=_MAXIMUM_CAMPAIGN_SEEDS)] = 3
    h1_task_count: Annotated[int, Field(ge=1, le=32)] = 3
    h1_synthesis_seed_count: Annotated[int, Field(ge=2, le=8)] = 2
    h3_fuzz_cases_per_candidate: Annotated[int, Field(ge=1, le=256)] = 12
    h4_maximum_candidates: Annotated[int, Field(ge=1, le=64)] = 12
    h4_improvement_threshold: Annotated[float, Field(gt=0.0)] = 0.12


class ArtifactReference(_SuiteModel):
    path: NonEmpty
    sha256: Sha256
    size_bytes: NonNegativeInt


class InputSnapshot(_SuiteModel):
    input_id: Literal["reference_package", "autopsy_diagnosis", "fabric_plan"]
    root: NonEmpty
    tree_sha256: Sha256
    file_count: PositiveInt


class ArtifactManifestEntry(_SuiteModel):
    path: NonEmpty
    sha256: Sha256
    size_bytes: NonNegativeInt
    kind: Literal["configuration", "input", "campaign_report", "raw_evidence"]


class ArtifactManifest(_SuiteModel):
    schema_version: Literal["sloforge.genesis.evaluation-suite-manifest/v1"] = (
        "sloforge.genesis.evaluation-suite-manifest/v1"
    )
    entries: tuple[ArtifactManifestEntry, ...]
    total_size_bytes: NonNegativeInt

    @model_validator(mode="after")
    def canonical_entries(self) -> Self:
        paths = tuple(item.path for item in self.entries)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("manifest paths must be sorted and unique")
        if self.total_size_bytes != sum(item.size_bytes for item in self.entries):
            raise ValueError("manifest byte total is not derived from entries")
        return self


class Metric(_SuiteModel):
    name: NonEmpty
    value: MetricValue
    unit: NonEmpty
    evidence_scope: NonEmpty


class CampaignRecord(_SuiteModel):
    campaign_id: Literal["core", "h1", "h2_h9", "h3", "h4", "h5", "h6", "h7", "h8"]
    hypothesis_ids: tuple[NonEmpty, ...]
    status: EvaluationStatus
    evidence_scope: NonEmpty
    hardware_backed_runs: NonNegativeInt
    report: ArtifactReference
    native_validator: NonEmpty
    limitations: tuple[NonEmpty, ...]


class HypothesisResult(_SuiteModel):
    hypothesis_id: Literal["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9"]
    statement: NonEmpty
    status: EvaluationStatus
    outcome: HypothesisOutcome
    source_campaign_id: NonEmpty
    evidence_scope: NonEmpty
    hardware_backed_runs: NonNegativeInt
    metrics: tuple[Metric, ...]
    limitations: tuple[NonEmpty, ...]


class GenesisEvaluationSuiteReport(_SuiteModel):
    schema_version: Literal["sloforge.genesis.evaluation-suite/v1"] = (
        "sloforge.genesis.evaluation-suite/v1"
    )
    configuration: EvaluationSuiteConfiguration
    input_snapshots: tuple[InputSnapshot, ...]
    artifact_manifest: ArtifactReference
    campaigns: tuple[CampaignRecord, ...]
    hypotheses: tuple[HypothesisResult, ...]
    hardware_backed_campaigns: NonNegativeInt
    synthetic_or_cpu_only_campaigns: PositiveInt
    universal_proof_claims: Literal[0] = 0
    report_path: NonEmpty

    @model_validator(mode="after")
    def complete_matrix(self) -> Self:
        if tuple(item.input_id for item in self.input_snapshots) != (
            "reference_package",
            "autopsy_diagnosis",
            "fabric_plan",
        ):
            raise ValueError("suite input snapshots are incomplete or non-canonical")
        if tuple(item.campaign_id for item in self.campaigns) != (
            "core",
            "h1",
            "h2_h9",
            "h3",
            "h4",
            "h5",
            "h6",
            "h7",
            "h8",
        ):
            raise ValueError("suite campaign order is incomplete or non-canonical")
        if tuple(item.hypothesis_id for item in self.hypotheses) != tuple(
            f"H{index}" for index in range(1, 10)
        ):
            raise ValueError("suite must report H1 through H9 in order")
        hardware = sum(item.hardware_backed_runs > 0 for item in self.campaigns)
        if self.hardware_backed_campaigns != hardware:
            raise ValueError("hardware campaign count is not derived")
        if self.synthetic_or_cpu_only_campaigns != len(self.campaigns) - hardware:
            raise ValueError("synthetic/CPU campaign count is not derived")
        return self


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_once(path: Path, value: BaseModel | dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_json(value) + b"\n")


def _reference(root: Path, path: Path) -> ArtifactReference:
    if path.is_symlink():
        raise EvaluationSuiteValidationError("campaign report must not be a symlink")
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise EvaluationSuiteValidationError("campaign report escapes suite root") from error
    if not resolved.is_file():
        raise EvaluationSuiteValidationError("campaign report must be a regular file")
    return ArtifactReference(
        path=relative.as_posix(),
        sha256=_sha256(resolved),
        size_bytes=resolved.stat().st_size,
    )


def _campaign_seeds(configuration: EvaluationSuiteConfiguration) -> tuple[int, ...]:
    return tuple(configuration.seed + index for index in range(configuration.campaign_seed_count))


def _h1_synthesis_seeds(configuration: EvaluationSuiteConfiguration) -> tuple[int, ...]:
    return tuple(
        configuration.seed + 2000 + index for index in range(configuration.h1_synthesis_seed_count)
    )


def _h1_configuration(configuration: EvaluationSuiteConfiguration) -> H1CampaignConfiguration:
    return H1CampaignConfiguration(
        grammar=GrammarConfiguration(
            seed=configuration.seed + 3000,
            count=configuration.h1_task_count,
            minimum_blocks=2,
            maximum_blocks=5,
            public_cases_per_task=2,
            hidden_cases_per_task=3,
        ),
        synthesis_seeds=_h1_synthesis_seeds(configuration),
    )


def _tree_files(path: Path) -> tuple[Path, ...]:
    files = (path,) if path.is_file() else tuple(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"input snapshot is empty: {path}")
    if any(item.is_symlink() for item in files):
        raise ValueError(f"input snapshot contains a symlink: {path}")
    return tuple(sorted(files, key=lambda item: item.relative_to(path.parent).as_posix()))


def _reject_tree_symlinks(path: Path) -> None:
    if path.is_symlink() or (path.is_dir() and any(item.is_symlink() for item in path.rglob("*"))):
        raise ValueError(f"input snapshot contains a symlink: {path}")


def _tree_digest(path: Path) -> str:
    records = [
        {
            "path": item.relative_to(path if path.is_dir() else path.parent).as_posix(),
            "sha256": _sha256(item),
            "size_bytes": item.stat().st_size,
        }
        for item in _tree_files(path)
    ]
    return hashlib.sha256(canonical_json({"files": records})).hexdigest()


def _snapshot_inputs(
    root: Path,
    *,
    reference_package: Path,
    diagnosis_path: Path,
    fabric_fixture: Path,
) -> tuple[tuple[InputSnapshot, ...], Path, Path, Path]:
    for source in (reference_package, diagnosis_path, fabric_fixture):
        if not source.exists():
            raise ValueError(f"suite input is unavailable or symlinked: {source}")
        _reject_tree_symlinks(source)
    reference_copy = root / "inputs/reference-package"
    shutil.copytree(
        reference_package,
        reference_copy,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    diagnosis_copy = root / "inputs/autopsy-diagnosis.json"
    fabric_copy = root / "inputs/physical-execution-plan-v1.json"
    diagnosis_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(diagnosis_path, diagnosis_copy)
    shutil.copyfile(fabric_fixture, fabric_copy)
    snapshots = tuple(
        InputSnapshot(
            input_id=input_id,  # type: ignore[arg-type]
            root=path.relative_to(root).as_posix(),
            tree_sha256=_tree_digest(path),
            file_count=len(_tree_files(path)),
        )
        for input_id, path in (
            ("reference_package", reference_copy),
            ("autopsy_diagnosis", diagnosis_copy),
            ("fabric_plan", fabric_copy),
        )
    )
    return snapshots, reference_copy, diagnosis_copy, fabric_copy


def _metric(name: str, value: MetricValue, unit: str, scope: str) -> Metric:
    return Metric(name=name, value=value, unit=unit, evidence_scope=scope)


def _core_record(root: Path, report: EvaluationResult) -> CampaignRecord:
    report_path = Path(report.report_path)
    validate_genesis_evaluation(report)
    return CampaignRecord(
        campaign_id="core",
        hypothesis_ids=(),
        status=EvaluationStatus.COMPLETED,
        evidence_scope="multi_seed_local_cpu_genesis_vertical_slice_and_host_timing",
        hardware_backed_runs=report.hardware_backed_runs,
        report=_reference(root, report_path),
        native_validator="sloforge.genesis.evaluation.validate_genesis_evaluation",
        limitations=(
            "The legacy core evaluation is supporting evidence; dedicated campaigns own H1-H9 conclusions.",
            "Host timings are local observations and are not GPU or distributed hardware evidence.",
        ),
    )


def _h1_records(root: Path, report: H1CampaignReport) -> tuple[CampaignRecord, HypothesisResult]:
    validate_h1_unseen_campaign(report)
    scope = "randomized_unseen_task_cpu_generated_runtime_with_public_and_hidden_cases"
    limitations = (
        report.hardware_validation_reason,
        "Task-level Wilson intervals do not establish behavior outside the typed task grammar.",
        "Synthesis-seed Wilson intervals are descriptive repeated deterministic attempts, not independent population samples.",
    )
    campaign = CampaignRecord(
        campaign_id="h1",
        hypothesis_ids=("H1",),
        status=EvaluationStatus.COMPLETED,
        evidence_scope=scope,
        hardware_backed_runs=0,
        report=_reference(root, root / "campaigns/h1/report.json"),
        native_validator="sloforge.genesis.evaluation_campaigns.unseen.validate_h1_unseen_campaign",
        limitations=limitations,
    )
    aggregate = report.aggregate
    supported = aggregate.valid_system_count == aggregate.task_count
    hypothesis = HypothesisResult(
        hypothesis_id="H1",
        statement="Genesis can synthesize correct baseline servers for unseen model tasks.",
        status=EvaluationStatus.COMPLETED,
        outcome=(
            HypothesisOutcome.SUPPORTED_IN_DECLARED_SCOPE
            if supported
            else HypothesisOutcome.NOT_SUPPORTED_IN_DECLARED_SCOPE
        ),
        source_campaign_id="h1",
        evidence_scope=scope,
        hardware_backed_runs=0,
        metrics=(
            _metric("task_count", aggregate.task_count, "unseen_tasks", scope),
            _metric("valid_system_rate", aggregate.valid_system_rate, "fraction", scope),
            _metric(
                "valid_system_95_wilson_low",
                aggregate.valid_system_interval.lower,
                "fraction",
                scope,
            ),
            _metric(
                "valid_system_95_wilson_high",
                aggregate.valid_system_interval.upper,
                "fraction",
                scope,
            ),
            _metric("exact_hidden_case_rate", aggregate.exact_hidden_case_rate, "fraction", scope),
            _metric(
                "human_authored_model_specific_lines",
                aggregate.human_authored_model_specific_lines,
                "lines",
                scope,
            ),
        ),
        limitations=limitations,
    )
    return campaign, hypothesis


def _whole_stack_records(
    root: Path, report: WholeStackCampaignReport
) -> tuple[CampaignRecord, HypothesisResult, HypothesisResult]:
    validate_whole_stack_campaign(report)
    scope = "deterministic_cpu_service_model_with_real_policy_state_lowering"
    campaign = CampaignRecord(
        campaign_id="h2_h9",
        hypothesis_ids=("H2", "H9"),
        status=EvaluationStatus.COMPLETED,
        evidence_scope=scope,
        hardware_backed_runs=report.hardware_backed_runs,
        report=_reference(root, root / "campaigns/whole-stack/report.json"),
        native_validator=(
            "sloforge.genesis.evaluation_campaigns.whole_stack.validate_whole_stack_campaign"
        ),
        limitations=report.limitations,
    )

    def result(
        hypothesis_id: Literal["H2", "H9"], statement: str, effect_name: str
    ) -> HypothesisResult:
        effect = report.h2_effect if hypothesis_id == "H2" else report.h9_effect
        conclusion = report.h2_conclusion if hypothesis_id == "H2" else report.h9_conclusion
        return HypothesisResult(
            hypothesis_id=hypothesis_id,
            statement=statement,
            status=EvaluationStatus.COMPLETED,
            outcome=(
                HypothesisOutcome.SUPPORTED_IN_DECLARED_SCOPE
                if conclusion == "supported_in_declared_synthetic_scope"
                else HypothesisOutcome.NOT_SUPPORTED_IN_DECLARED_SCOPE
            ),
            source_campaign_id="h2_h9",
            evidence_scope=scope,
            hardware_backed_runs=0,
            metrics=(
                _metric(effect_name, effect.mean_difference, "modeled_objective_units", scope),
                _metric(
                    "paired_95_ci_low", effect.confidence_low, "modeled_objective_units", scope
                ),
                _metric(
                    "paired_95_ci_high", effect.confidence_high, "modeled_objective_units", scope
                ),
                _metric("seed_count", len(report.seeds), "paired_seeds", scope),
            ),
            limitations=report.limitations,
        )

    return (
        campaign,
        result(
            "H2",
            "Whole-stack synthesis outperforms configuration-only optimization in selected workloads.",
            "configuration_only_minus_genesis",
        ),
        result(
            "H9",
            "Cross-layer candidates can outperform isolated optimization.",
            "best_single_layer_minus_genesis",
        ),
    )


def _h3_records(root: Path, report: H3CampaignReport) -> tuple[CampaignRecord, HypothesisResult]:
    validate_cegis_campaign(root / "campaigns/h3")
    full = next(
        item for item in report.aggregates if item.strategy is VerificationStrategy.FULL_CEGIS
    )
    tests_only = next(
        item for item in report.aggregates if item.strategy is VerificationStrategy.TESTS_ONLY
    )
    fuzz_only = next(
        item for item in report.aggregates if item.strategy is VerificationStrategy.FUZZING_ONLY
    )
    model_check_only = next(
        item for item in report.aggregates if item.strategy is VerificationStrategy.MODEL_CHECK_ONLY
    )
    comparator_escapes = (
        tests_only.escaped_faults,
        fuzz_only.escaped_faults,
        model_check_only.escaped_faults,
    )
    strictly_better_all = all(full.escaped_faults < value for value in comparator_escapes)
    no_worse_with_some_reduction = all(
        full.escaped_faults <= value for value in comparator_escapes
    ) and any(full.escaped_faults < value for value in comparator_escapes)
    scope = report.scope.evidence_scope
    campaign = CampaignRecord(
        campaign_id="h3",
        hypothesis_ids=("H3",),
        status=EvaluationStatus.COMPLETED,
        evidence_scope=scope,
        hardware_backed_runs=0,
        report=_reference(root, root / "campaigns/h3/report.json"),
        native_validator="sloforge.genesis.evaluation_campaigns.cegis.validate_cegis_campaign",
        limitations=report.limitations,
    )
    hypothesis = HypothesisResult(
        hypothesis_id="H3",
        statement="Counterexample-guided synthesis reduces escaped failures.",
        status=EvaluationStatus.COMPLETED,
        outcome=(
            HypothesisOutcome.SUPPORTED_IN_DECLARED_SCOPE
            if strictly_better_all and full.learned_constraint_reuses > 0
            else HypothesisOutcome.MIXED_IN_DECLARED_SCOPE
            if no_worse_with_some_reduction and full.learned_constraint_reuses > 0
            else HypothesisOutcome.NOT_SUPPORTED_IN_DECLARED_SCOPE
        ),
        source_campaign_id="h3",
        evidence_scope=scope,
        hardware_backed_runs=0,
        metrics=(
            _metric("fault_instances", full.fault_instances, "candidate_fault_instances", scope),
            _metric("escaped_faults", full.escaped_faults, "candidate_fault_instances", scope),
            _metric(
                "tests_only_escaped_faults",
                tests_only.escaped_faults,
                "candidate_fault_instances",
                scope,
            ),
            _metric(
                "fuzzing_only_escaped_faults",
                fuzz_only.escaped_faults,
                "candidate_fault_instances",
                scope,
            ),
            _metric(
                "model_check_only_escaped_faults",
                model_check_only.escaped_faults,
                "candidate_fault_instances",
                scope,
            ),
            _metric("learned_constraint_reuses", full.learned_constraint_reuses, "reuses", scope),
            _metric(
                "median_minimized_counterexample_events",
                full.median_minimized_counterexample_events or 0.0,
                "events",
                scope,
            ),
        ),
        limitations=report.limitations,
    )
    return campaign, hypothesis


def _h4_records(
    root: Path, report: AutopsyCampaignReport
) -> tuple[CampaignRecord, HypothesisResult]:
    validate_autopsy_guided_campaign(report)
    scope = report.scope.evidence_scope
    comparisons_positive = all(
        item.candidates_evaluated_reduction >= 0
        and item.invalid_candidate_reduction >= 0
        and item.mean_time_to_improvement_reduction is not None
        and item.mean_time_to_improvement_reduction > 0
        and item.mean_final_objective_delta is not None
        and item.mean_final_objective_delta > 0
        for item in report.guided_deltas
    )
    campaign = CampaignRecord(
        campaign_id="h4",
        hypothesis_ids=("H4",),
        status=EvaluationStatus.COMPLETED,
        evidence_scope=scope,
        hardware_backed_runs=0,
        report=_reference(root, Path(report.report_path)),
        native_validator=(
            "sloforge.genesis.evaluation_campaigns.autopsy.validate_autopsy_guided_campaign"
        ),
        limitations=report.limitations,
    )
    metrics = tuple(
        metric
        for item in report.guided_deltas
        for metric in (
            _metric(
                f"{item.baseline.value}_candidate_reduction",
                item.candidates_evaluated_reduction,
                "candidates",
                scope,
            ),
            _metric(
                f"{item.baseline.value}_invalid_candidate_reduction",
                item.invalid_candidate_reduction,
                "candidates",
                scope,
            ),
            _metric(
                f"{item.baseline.value}_time_reduction",
                (
                    item.mean_time_to_improvement_reduction
                    if item.mean_time_to_improvement_reduction is not None
                    else "not_available"
                ),
                "synthetic_seconds",
                scope,
            ),
            _metric(
                f"{item.baseline.value}_final_objective_delta",
                (
                    item.mean_final_objective_delta
                    if item.mean_final_objective_delta is not None
                    else "not_available"
                ),
                "normalized_utility_points",
                scope,
            ),
        )
    )
    hypothesis = HypothesisResult(
        hypothesis_id="H4",
        statement="Autopsy-guided mutation improves search efficiency.",
        status=EvaluationStatus.COMPLETED,
        outcome=(
            HypothesisOutcome.SUPPORTED_IN_DECLARED_SCOPE
            if comparisons_positive
            else HypothesisOutcome.MIXED_IN_DECLARED_SCOPE
        ),
        source_campaign_id="h4",
        evidence_scope=scope,
        hardware_backed_runs=0,
        metrics=metrics,
        limitations=report.limitations,
    )
    return campaign, hypothesis


def _h5_records(
    root: Path, report: H5LineageCampaignReport
) -> tuple[CampaignRecord, HypothesisResult]:
    validate_h5_lineage_campaign(root / "campaigns/h5", report)
    scope = report.metric_scope
    campaign = CampaignRecord(
        campaign_id="h5",
        hypothesis_ids=("H5",),
        status=EvaluationStatus.COMPLETED,
        evidence_scope=scope,
        hardware_backed_runs=report.hardware_backed_runs,
        report=_reference(root, root / "campaigns/h5/report.json"),
        native_validator="sloforge.genesis.evaluation_campaigns.lineage.validate_h5_lineage_campaign",
        limitations=report.limitations,
    )
    summary = report.effect_summary
    hypothesis = HypothesisResult(
        hypothesis_id="H5",
        statement="Lineage transfer reduces optimization cost on related tasks.",
        status=EvaluationStatus.COMPLETED,
        outcome=HypothesisOutcome.SUPPORTED_IN_DECLARED_SCOPE,
        source_campaign_id="h5",
        evidence_scope=scope,
        hardware_backed_runs=0,
        metrics=(
            _metric(
                "median_related_candidate_units_saved_to_first_improved",
                summary.median_related_candidate_units_saved_to_first_improved,
                "candidate_units",
                scope,
            ),
            _metric(
                "median_related_time_units_saved_to_first_improved",
                summary.median_related_time_units_saved_to_first_improved,
                "synthetic_time_units",
                scope,
            ),
            _metric(
                "invalidation_negative_transfers_avoided",
                summary.total_invalidation_negative_transfers_avoided,
                "transfer_events",
                scope,
            ),
        ),
        limitations=report.limitations,
    )
    return campaign, hypothesis


def _h8_records(
    root: Path, report: RedTeamCampaignReport
) -> tuple[CampaignRecord, HypothesisResult]:
    validate_redteam_campaign(report)
    scope = report.scope.evidence_scope
    campaign = CampaignRecord(
        campaign_id="h8",
        hypothesis_ids=("H8",),
        status=EvaluationStatus.COMPLETED,
        evidence_scope=scope,
        hardware_backed_runs=report.scope.hardware_backed_runs,
        report=_reference(root, root / "campaigns/h8/report.json"),
        native_validator=(
            "sloforge.genesis.evaluation_campaigns.redteam.validate_redteam_campaign"
        ),
        limitations=report.limitations,
    )
    hypothesis = HypothesisResult(
        hypothesis_id="H8",
        statement="The executable red team discovers violations not found by normal tests.",
        status=EvaluationStatus.COMPLETED,
        outcome=(
            HypothesisOutcome.SUPPORTED_IN_DECLARED_SCOPE
            if report.conclusion == "supported_in_declared_fixture_scope"
            else HypothesisOutcome.NOT_SUPPORTED_IN_DECLARED_SCOPE
        ),
        source_campaign_id="h8",
        evidence_scope=scope,
        hardware_backed_runs=0,
        metrics=(
            _metric(
                "redteam_only_violation_contracts",
                len(report.redteam_only_violation_contracts),
                "contract_families",
                scope,
            ),
            _metric("converted_regressions", report.converted_regressions, "regressions", scope),
            _metric("reproduced_regressions", report.reproduced_regressions, "regressions", scope),
        ),
        limitations=report.limitations,
    )
    return campaign, hypothesis


CapsuleValidator = Callable[[Path], CapsuleValidationReport]


def _capsule_validator(references: tuple[CapsuleReference, ...]) -> CapsuleValidator:
    values: dict[Path, tuple[GenesisCapsule, Path, ValidationContext]] = {}
    for reference in references:
        manifest_path = Path(reference.path).resolve(strict=True)
        capsule_root = manifest_path.parent.parent
        context_path = capsule_root.with_name(f"{capsule_root.name}.validation-context.json")
        capsule = load_capsule(manifest_path)
        context = ValidationContext.model_validate_json(context_path.read_bytes(), strict=True)
        values[manifest_path] = (capsule, capsule_root, context)

    def validate(path: Path) -> CapsuleValidationReport:
        capsule, root, context = values[path.resolve(strict=True)]
        return validate_capsule(capsule, root, context)

    return validate


def _core_capsule_references(core: EvaluationResult) -> tuple[CapsuleReference, CapsuleReference]:
    first_report = Path(core.report_path).parent / "runs" / str(core.seeds[0])
    demo = GenesisDemoResult.model_validate_json(
        (first_report / "GENESIS_DEMO_REPORT.json").read_bytes(), strict=True
    )
    champion_manifest = Path(demo.capsule_path)
    challenger_manifests = tuple(
        sorted(
            (Path(demo.output_directory) / "evolution/challenger-capsule/manifests").glob("*.json")
        )
    )
    if len(challenger_manifests) != 1:
        raise EvaluationSuiteValidationError("core run does not expose one challenger capsule")
    challenger_manifest = challenger_manifests[0]
    challenger = load_capsule(challenger_manifest)
    if challenger.capsule_digest is None:
        raise EvaluationSuiteValidationError("challenger capsule is not sealed")
    return (
        CapsuleReference(
            capsule_id="suite-core-champion",
            capsule_digest=demo.capsule_digest,
            genome_hash=demo.accepted_genome_hash,
            path=str(champion_manifest.resolve()),
        ),
        CapsuleReference(
            capsule_id="suite-core-challenger",
            capsule_digest=challenger.capsule_digest.value,
            genome_hash=challenger.identity.candidate_genome_hash.value,
            path=str(challenger_manifest.resolve()),
        ),
    )


def _run_h6_hook(root: Path, _core: EvaluationResult, seeds: tuple[int, ...]) -> Path:
    run_capsule_attack_campaign(root / "campaigns/h6", seeds=seeds)
    return root / "campaigns/h6/report.json"


def _validate_h6_hook(root: Path, report_path: Path) -> tuple[CampaignRecord, HypothesisResult]:
    report = validate_capsule_attack_campaign(report_path)
    scope = report.scope.evidence_scope
    campaign = CampaignRecord(
        campaign_id="h6",
        hypothesis_ids=("H6",),
        status=EvaluationStatus.COMPLETED,
        evidence_scope=scope,
        hardware_backed_runs=report.scope.hardware_backed_runs,
        report=_reference(root, report_path),
        native_validator=(
            "sloforge.genesis.evaluation_campaigns.capsule_attacks.validate_capsule_attack_campaign"
        ),
        limitations=report.limitations,
    )
    hypothesis = HypothesisResult(
        hypothesis_id="H6",
        statement="Proof-carrying promotion catches unsafe or incompatible generated artifacts.",
        status=EvaluationStatus.COMPLETED,
        outcome=(
            HypothesisOutcome.SUPPORTED_IN_DECLARED_SCOPE
            if report.conclusion == "supported_in_declared_fixture_scope"
            else HypothesisOutcome.NOT_SUPPORTED_IN_DECLARED_SCOPE
        ),
        source_campaign_id="h6",
        evidence_scope=scope,
        hardware_backed_runs=0,
        metrics=(
            _metric("attacks_executed", report.attack_count, "attack_cases", scope),
            _metric("attacks_rejected", report.rejected_attack_count, "attack_cases", scope),
            _metric(
                "validator_issue_code_coverage",
                len(report.issue_code_coverage),
                "distinct_issue_codes",
                scope,
            ),
        ),
        limitations=report.limitations,
    )
    return campaign, hypothesis


def _run_h7_hook(root: Path, core: EvaluationResult, seeds: tuple[int, ...]) -> Path:
    from sloforge.genesis.evaluation_campaigns.evolution import run_evolution_campaign

    champion, challenger = _core_capsule_references(core)
    report = run_evolution_campaign(
        root / "campaigns/h7",
        champion=champion,
        challenger=challenger,
        capsule_validator=_capsule_validator((champion, challenger)),
        seeds=seeds,
        rollback_seeds=(seeds[-1],),
    )
    return Path(report.report_path)


def _validate_h7_hook(root: Path, report_path: Path) -> tuple[CampaignRecord, HypothesisResult]:
    from sloforge.genesis.evaluation_campaigns.evolution import (
        AdaptationStrategy,
        EvolutionCampaignReport,
        validate_evolution_campaign,
    )

    preliminary = EvolutionCampaignReport.model_validate_json(report_path.read_bytes(), strict=True)
    references = tuple(
        item
        for evidence in preliminary.genesis_evidence
        for item in (evidence.champion, evidence.challenger)
    )
    unique = tuple({item.path: item for item in references}.values())
    report = validate_evolution_campaign(
        report_path,
        capsule_validator=_capsule_validator(unique),
    )
    scope = f"{report.logical_time_scope}; {report.local_runtime_scope}"
    genesis = next(
        item for item in report.aggregates if item.strategy is AdaptationStrategy.GENESIS_EVOLUTION
    )
    no_adaptation = next(
        item for item in report.aggregates if item.strategy is AdaptationStrategy.NO_ADAPTATION
    )
    supported = (
        genesis.restoration_rate > no_adaptation.restoration_rate
        and genesis.total_interrupted_streams == 0
    )
    campaign = CampaignRecord(
        campaign_id="h7",
        hypothesis_ids=("H7",),
        status=EvaluationStatus.COMPLETED,
        evidence_scope=scope,
        hardware_backed_runs=report.actual_hardware_experiments,
        report=_reference(root, report_path),
        native_validator=(
            "sloforge.genesis.evaluation_campaigns.evolution.validate_evolution_campaign"
        ),
        limitations=report.limitations,
    )
    hypothesis = HypothesisResult(
        hypothesis_id="H7",
        statement="Continuous evolution adapts to workload and hardware drift better than static deployment.",
        status=EvaluationStatus.COMPLETED,
        outcome=(
            HypothesisOutcome.SUPPORTED_IN_DECLARED_SCOPE
            if supported
            else HypothesisOutcome.NOT_SUPPORTED_IN_DECLARED_SCOPE
        ),
        source_campaign_id="h7",
        evidence_scope=scope,
        hardware_backed_runs=0,
        metrics=(
            _metric("genesis_restoration_rate", genesis.restoration_rate, "fraction", scope),
            _metric("static_restoration_rate", no_adaptation.restoration_rate, "fraction", scope),
            _metric("interrupted_streams", genesis.total_interrupted_streams, "streams", scope),
            _metric("rollbacks", genesis.total_rollbacks, "controller_events", scope),
        ),
        limitations=report.limitations,
    )
    return campaign, hypothesis


def _manifest_kind(
    relative: str,
) -> Literal["configuration", "input", "campaign_report", "raw_evidence"]:
    if relative == _CONFIGURATION_NAME:
        return "configuration"
    if relative.startswith("inputs/"):
        return "input"
    if relative.endswith("/report.json") or relative.endswith("/evaluation.json"):
        return "campaign_report"
    return "raw_evidence"


def _artifact_files(root: Path) -> tuple[Path, ...]:
    excluded = {_MANIFEST_NAME, _REPORT_NAME}
    paths = tuple(
        path for path in root.rglob("*") if path.relative_to(root).as_posix() not in excluded
    )
    if any(path.is_symlink() for path in paths):
        raise EvaluationSuiteValidationError("suite artifact tree contains a symlink")
    return tuple(
        sorted(
            (path for path in paths if path.is_file()),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    )


def _build_manifest(root: Path) -> ArtifactManifest:
    entries = tuple(
        ArtifactManifestEntry(
            path=path.relative_to(root).as_posix(),
            sha256=_sha256(path),
            size_bytes=path.stat().st_size,
            kind=_manifest_kind(path.relative_to(root).as_posix()),
        )
        for path in _artifact_files(root)
    )
    return ArtifactManifest(
        entries=entries,
        total_size_bytes=sum(item.size_bytes for item in entries),
    )


def _load_native_records(
    root: Path, configuration: EvaluationSuiteConfiguration
) -> tuple[tuple[CampaignRecord, ...], tuple[HypothesisResult, ...]]:
    core_path = root / "campaigns/core/evaluation.json"
    core = EvaluationResult.model_validate_json(core_path.read_bytes(), strict=True)
    if (
        core.seed != configuration.seed
        or core.run_count != configuration.core_run_count
        or core.seeds
        != tuple(configuration.seed + index for index in range(configuration.core_run_count))
    ):
        raise EvaluationSuiteValidationError("core campaign seeds differ from suite configuration")
    h1 = H1CampaignReport.model_validate_json(
        (root / "campaigns/h1/report.json").read_bytes(), strict=True
    )
    if h1.configuration != _h1_configuration(configuration):
        raise EvaluationSuiteValidationError("H1 configuration differs from suite configuration")
    whole = WholeStackCampaignReport.model_validate_json(
        (root / "campaigns/whole-stack/report.json").read_bytes(), strict=True
    )
    h3 = H3CampaignReport.model_validate_json(
        (root / "campaigns/h3/report.json").read_bytes(), strict=True
    )
    h4 = AutopsyCampaignReport.model_validate_json(
        (root / "campaigns/h4/report.json").read_bytes(), strict=True
    )
    h5 = H5LineageCampaignReport.model_validate_json(
        (root / "campaigns/h5/report.json").read_bytes(), strict=True
    )
    h8 = RedTeamCampaignReport.model_validate_json(
        (root / "campaigns/h8/report.json").read_bytes(), strict=True
    )
    seeds = _campaign_seeds(configuration)
    if whole.seeds != seeds or h8.seeds != seeds or h5.seeds != seeds:
        raise EvaluationSuiteValidationError("campaign seeds differ from suite configuration")
    if h3.base_seed != configuration.seed or len(h3.run_seeds) != len(seeds):
        raise EvaluationSuiteValidationError("H3 seeds differ from suite configuration")
    if h4.campaign_seed != configuration.seed or len(h4.run_seeds) != len(seeds):
        raise EvaluationSuiteValidationError("H4 seeds differ from suite configuration")
    core_record = _core_record(root, core)
    h1_record, h1_hypothesis = _h1_records(root, h1)
    whole_record, h2_hypothesis, h9_hypothesis = _whole_stack_records(root, whole)
    h3_record, h3_hypothesis = _h3_records(root, h3)
    h4_record, h4_hypothesis = _h4_records(root, h4)
    h5_record, h5_hypothesis = _h5_records(root, h5)
    h6_record, h6_hypothesis = _validate_h6_hook(root, root / "campaigns/h6/report.json")
    h7_record, h7_hypothesis = _validate_h7_hook(root, root / "campaigns/h7/report.json")
    h8_record, h8_hypothesis = _h8_records(root, h8)
    return (
        (
            core_record,
            h1_record,
            whole_record,
            h3_record,
            h4_record,
            h5_record,
            h6_record,
            h7_record,
            h8_record,
        ),
        (
            h1_hypothesis,
            h2_hypothesis,
            h3_hypothesis,
            h4_hypothesis,
            h5_hypothesis,
            h6_hypothesis,
            h7_hypothesis,
            h8_hypothesis,
            h9_hypothesis,
        ),
    )


def run_genesis_evaluation_suite(
    output: Path,
    *,
    configuration: EvaluationSuiteConfiguration | None = None,
    reference_package: Path | None = None,
    diagnosis_path: Path | None = None,
    fabric_fixture: Path | None = None,
) -> GenesisEvaluationSuiteReport:
    """Run the complete local/synthetic suite without claiming unavailable hardware evidence."""

    configuration = configuration or EvaluationSuiteConfiguration()
    if output.exists() and (output.is_symlink() or not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"suite output must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[3]
    reference_source = reference_package or repository / "models/reference_tasks/hybrid_decoder"
    diagnosis_source = diagnosis_path or repository / "tests/fixtures/autopsy/diagnosis-v1.json"
    fabric_source = (
        fabric_fixture or repository / "tests/fixtures/fabric/physical-execution-plan-v1.json"
    )
    snapshots, reference_copy, diagnosis_copy, fabric_copy = _snapshot_inputs(
        output,
        reference_package=reference_source,
        diagnosis_path=diagnosis_source,
        fabric_fixture=fabric_source,
    )
    _write_once(output / _CONFIGURATION_NAME, configuration)
    core = run_genesis_evaluation(
        output / "campaigns/core",
        seed=configuration.seed,
        count=configuration.core_run_count,
    )
    run_h1_unseen_campaign(_h1_configuration(configuration), output / "campaigns/h1")
    seeds = _campaign_seeds(configuration)
    run_whole_stack_campaign(
        output / "campaigns/whole-stack",
        seeds=seeds,
        reference_package=reference_copy,
        fabric_fixture=fabric_copy,
    )
    run_cegis_campaign(
        output / "campaigns/h3",
        base_seed=configuration.seed,
        seed_count=configuration.campaign_seed_count,
        fuzz_cases_per_candidate=configuration.h3_fuzz_cases_per_candidate,
    )
    run_autopsy_guided_campaign(
        output / "campaigns/h4",
        diagnosis_path=diagnosis_copy,
        seed=configuration.seed,
        count=configuration.campaign_seed_count,
        maximum_candidates=configuration.h4_maximum_candidates,
        improvement_threshold=configuration.h4_improvement_threshold,
    )
    run_h5_lineage_campaign(
        output / "campaigns/h5",
        seed=seeds[0],
        count=configuration.campaign_seed_count,
    )
    _run_h6_hook(output, core, seeds)
    _run_h7_hook(output, core, seeds)
    run_redteam_campaign(output / "campaigns/h8", seeds=seeds)
    campaigns, hypotheses = _load_native_records(output, configuration)
    manifest = _build_manifest(output)
    manifest_path = output / _MANIFEST_NAME
    _write_once(manifest_path, manifest)
    report_path = output / _REPORT_NAME
    report = GenesisEvaluationSuiteReport(
        configuration=configuration,
        input_snapshots=snapshots,
        artifact_manifest=_reference(output, manifest_path),
        campaigns=campaigns,
        hypotheses=hypotheses,
        hardware_backed_campaigns=sum(item.hardware_backed_runs > 0 for item in campaigns),
        synthetic_or_cpu_only_campaigns=sum(item.hardware_backed_runs == 0 for item in campaigns),
        report_path=str(report_path.resolve()),
    )
    _write_once(report_path, report)
    return validate_genesis_evaluation_suite(report_path)


def _validate_snapshot(root: Path, snapshot: InputSnapshot) -> None:
    unresolved = root / snapshot.root
    if unresolved.is_symlink():
        raise EvaluationSuiteValidationError("input snapshot is symlinked")
    path = unresolved.resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise EvaluationSuiteValidationError("input snapshot escapes suite root") from error
    if len(_tree_files(path)) != snapshot.file_count:
        raise EvaluationSuiteValidationError("input snapshot contents changed")
    if _tree_digest(path) != snapshot.tree_sha256:
        raise EvaluationSuiteValidationError("input snapshot digest changed")


def validate_genesis_evaluation_suite(report_path: Path) -> GenesisEvaluationSuiteReport:
    """Reopen the root manifest and invoke every campaign's independent validator."""

    if report_path.is_symlink() or not report_path.is_file():
        raise EvaluationSuiteValidationError("suite report is missing or symlinked")
    try:
        report = GenesisEvaluationSuiteReport.model_validate_json(
            report_path.read_bytes(), strict=True
        )
    except ValueError as error:
        raise EvaluationSuiteValidationError("suite report schema validation failed") from error
    if Path(report.report_path).resolve() != report_path.resolve():
        raise EvaluationSuiteValidationError("suite report path is not self-consistent")
    root = report_path.resolve().parent
    configuration_path = root / _CONFIGURATION_NAME
    configuration = EvaluationSuiteConfiguration.model_validate_json(
        configuration_path.read_bytes(), strict=True
    )
    if configuration != report.configuration:
        raise EvaluationSuiteValidationError("persisted suite configuration changed")
    manifest_path = (root / report.artifact_manifest.path).resolve(strict=True)
    if (
        manifest_path != root / _MANIFEST_NAME
        or _sha256(manifest_path) != report.artifact_manifest.sha256
        or manifest_path.stat().st_size != report.artifact_manifest.size_bytes
    ):
        raise EvaluationSuiteValidationError("suite artifact manifest changed")
    manifest = ArtifactManifest.model_validate_json(manifest_path.read_bytes(), strict=True)
    expected_manifest = _build_manifest(root)
    if manifest != expected_manifest:
        raise EvaluationSuiteValidationError("suite artifact manifest does not match the tree")
    for snapshot in report.input_snapshots:
        _validate_snapshot(root, snapshot)
    campaigns, hypotheses = _load_native_records(root, configuration)
    if campaigns != report.campaigns or hypotheses != report.hypotheses:
        raise EvaluationSuiteValidationError("suite claims are not derived from native campaigns")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--seed", type=int, default=73129)
    run_parser.add_argument("--core-runs", type=int, default=2)
    run_parser.add_argument("--campaign-seeds", type=int, default=3)
    run_parser.add_argument("--h1-tasks", type=int, default=3)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "run":
        report = run_genesis_evaluation_suite(
            arguments.output,
            configuration=EvaluationSuiteConfiguration(
                seed=arguments.seed,
                core_run_count=arguments.core_runs,
                campaign_seed_count=arguments.campaign_seeds,
                h1_task_count=arguments.h1_tasks,
            ),
        )
    else:
        report = validate_genesis_evaluation_suite(arguments.report)
    print(report.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ArtifactManifest",
    "EvaluationSuiteConfiguration",
    "EvaluationSuiteValidationError",
    "GenesisEvaluationSuiteReport",
    "HypothesisOutcome",
    "run_genesis_evaluation_suite",
    "validate_genesis_evaluation_suite",
]
