"""Compose the browser UI bundle from independently persisted Genesis artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Annotated, Any, Final, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    ValidationError,
    model_validator,
)

from sloforge.genesis.capsule import GenesisCapsule, calculate_capsule_digest
from sloforge.genesis.capsule.models import RawBenchmarkSamples
from sloforge.genesis.evolution import EvolutionSnapshot
from sloforge.genesis.ir import (
    Candidate,
    CandidateFailureState,
    Counterexample,
    InferenceGenome,
    canonical_hash,
    canonical_json,
    load_candidate,
    load_counterexample,
    load_inference_genome,
)
from sloforge.genesis.search import CandidateDesign

GENESIS_UI_BUNDLE_VERSION: Final[Literal["sloforge.genesis.ui-bundle/v1"]] = (
    "sloforge.genesis.ui-bundle/v1"
)
_MAX_DOCUMENT_BYTES = 64 * 1024 * 1024
_SHA256 = r"^[a-f0-9]{64}$"
_NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_DigestString = Annotated[str, StringConstraints(pattern=_SHA256)]


class GenesisUiBundleError(RuntimeError):
    """Raised when persisted evidence cannot form a consistent UI bundle."""


class _BundleModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        allow_inf_nan=False,
    )


class DemoSummary(_BundleModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    seed: int = Field(ge=0)
    output_directory: _NonEmpty
    package_id: _NonEmpty
    operator_count: int = Field(ge=0)
    state_field_count: int = Field(ge=0)
    baseline_genome_hash: _DigestString
    accepted_candidate_id: _NonEmpty
    accepted_genome_hash: _DigestString
    cross_layer_accepted: bool
    rejected_candidate_ids: tuple[_NonEmpty, ...]
    minimized_counterexample_ids: tuple[_NonEmpty, ...]
    learned_constraint_ids: tuple[_NonEmpty, ...]
    runtime_differential_passed: bool
    capsule_path: _NonEmpty
    capsule_digest: _DigestString
    capsule_promotion_eligible: bool
    redteam_finding_count: int = Field(ge=0)
    redteam_replayed_count: int = Field(ge=0)
    kernel_candidate_count: int = Field(ge=0)
    kernel_speedup_claim_count: int = Field(ge=0)
    evolution_promoted: bool
    active_stream_preserved: bool
    physical_degradation_triggered: bool
    hardware_backed: bool
    report_path: _NonEmpty
    ui_bundle_path: _NonEmpty


class CandidateBundle(_BundleModel):
    candidate: Candidate
    design: CandidateDesign

    @model_validator(mode="after")
    def bind_design_to_candidate(self) -> Self:
        if self.candidate.candidate_id != self.design.candidate_id:
            raise ValueError("candidate and design identifiers differ")
        if self.candidate.parent_candidate_ids != self.design.parent_candidate_ids:
            raise ValueError("candidate and design ancestry differs")
        mutation_ids = tuple(item.transformation_id for item in self.design.mutations)
        if self.candidate.transformation_ids != mutation_ids:
            raise ValueError("candidate transformations differ from its design")
        return self


class LineageCase(_BundleModel):
    lineage_seed_ids: tuple[_NonEmpty, ...]
    lineage_seed_count: int = Field(ge=0)
    unseeded_count: int = Field(ge=0)
    reverification_required: bool

    @model_validator(mode="after")
    def count_seed_ids(self) -> Self:
        if self.lineage_seed_count != len(self.lineage_seed_ids):
            raise ValueError("lineage seed count differs from seed identifiers")
        return self


class LineageCases(_BundleModel):
    empty_lineage: LineageCase
    unrelated_lineage: LineageCase
    related_lineage: LineageCase
    stale_dependency_before_invalidation: LineageCase
    stale_dependency_after_invalidation: LineageCase


class LineageTransferReport(_BundleModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    seed: int = Field(ge=0)
    scope: _NonEmpty
    cases: LineageCases
    affected_evidence_count: int = Field(ge=0)
    related_seed_retrieved: bool
    stale_seed_suppressed_after_invalidation: bool
    performance_hypothesis_evaluated: bool

    @model_validator(mode="after")
    def outcomes_follow_cases(self) -> Self:
        related = self.cases.related_lineage.lineage_seed_count > 0
        stale_suppressed = (
            self.cases.stale_dependency_before_invalidation.lineage_seed_count > 0
            and self.cases.stale_dependency_after_invalidation.lineage_seed_count == 0
        )
        if self.related_seed_retrieved != related:
            raise ValueError("related-seed outcome differs from the related lineage case")
        if self.stale_seed_suppressed_after_invalidation != stale_suppressed:
            raise ValueError("invalidation outcome differs from stale lineage cases")
        return self


class GenesisUiBundle(_BundleModel):
    artifact_type: Literal["sloforge.genesis.ui-bundle/v1"] = GENESIS_UI_BUNDLE_VERSION
    summary: DemoSummary
    genome: InferenceGenome
    candidates: tuple[CandidateBundle, ...]
    counterexamples: tuple[Counterexample, ...]
    capsule: GenesisCapsule
    benchmark_definition: dict[str, JsonValue]
    baseline_samples: RawBenchmarkSamples
    candidate_samples: RawBenchmarkSamples
    evolution: EvolutionSnapshot
    lineage: LineageTransferReport

    @model_validator(mode="after")
    def cross_artifact_identities(self) -> Self:
        candidates = {item.candidate.candidate_id: item.candidate for item in self.candidates}
        accepted = candidates.get(self.summary.accepted_candidate_id)
        if accepted is None:
            raise ValueError("accepted candidate is absent from candidate artifacts")
        if accepted.genome_hash.value != self.summary.accepted_genome_hash:
            raise ValueError("accepted candidate genome hash differs from the demo summary")
        if canonical_hash(self.genome) != self.summary.accepted_genome_hash:
            raise ValueError("accepted InferenceGenome content hash differs from the demo summary")
        for rejected_id in self.summary.rejected_candidate_ids:
            rejected = candidates.get(rejected_id)
            if rejected is None or not isinstance(rejected.state, CandidateFailureState):
                raise ValueError("rejected candidate lacks a terminal failure artifact")
        if self.capsule.capsule_digest is None:
            raise ValueError("UI bundle requires a sealed capsule")
        if self.capsule.capsule_digest.value != self.summary.capsule_digest:
            raise ValueError("capsule digest differs from the demo summary")
        if self.capsule.identity.candidate_genome_hash.value != self.summary.accepted_genome_hash:
            raise ValueError("capsule identity differs from the accepted genome")
        counterexamples = {item.counterexample_id: item for item in self.counterexamples}
        for counterexample_id in self.summary.minimized_counterexample_ids:
            item = counterexamples.get(counterexample_id)
            if item is None or not item.minimized:
                raise ValueError("summary references a missing or non-minimized counterexample")
        benchmark = self.capsule.benchmarks[0] if len(self.capsule.benchmarks) == 1 else None
        if benchmark is None:
            raise ValueError("UI bundle requires exactly one capsule benchmark")
        if self.benchmark_definition.get("benchmark_id") != benchmark.benchmark_id:
            raise ValueError("benchmark definition identifier differs from capsule evidence")
        if self.benchmark_definition.get("hardware_backed") != self.summary.hardware_backed:
            raise ValueError("benchmark hardware mode differs from the demo summary")
        if len(self.candidate_samples.samples) != benchmark.sample_count:
            raise ValueError("candidate raw sample count differs from capsule evidence")
        if len(self.baseline_samples.samples) != benchmark.repetitions:
            raise ValueError("baseline raw sample count differs from capsule evidence")
        for samples in (self.baseline_samples, self.candidate_samples):
            if samples.workload_fingerprint != benchmark.workload_fingerprint:
                raise ValueError("raw sample workload fingerprint differs from capsule evidence")
            if samples.hardware_fingerprint != benchmark.hardware_fingerprint:
                raise ValueError("raw sample hardware fingerprint differs from capsule evidence")
        if self.evolution.seed != self.summary.seed:
            raise ValueError("evolution and demo seeds differ")
        if self.lineage.seed != self.summary.seed:
            raise ValueError("lineage and demo seeds differ")
        if self.summary.evolution_promoted and not any(
            item.action == "promote" for item in self.evolution.audit
        ):
            raise ValueError("promotion summary lacks a controller audit record")
        if self.summary.physical_degradation_triggered and (
            self.evolution.active_trigger is None
            or self.evolution.active_trigger.value != "fabric_degradation"
        ):
            raise ValueError("physical degradation summary differs from the controller trigger")
        previous = self.evolution.previous_champion
        if previous is None or previous.capsule_digest != self.summary.capsule_digest:
            raise ValueError("evolution previous champion differs from the original capsule")
        if self.summary.active_stream_preserved and not any(
            item.capsule_id == previous.capsule_id for item in self.evolution.active_streams
        ):
            raise ValueError("active-stream preservation lacks a pinned stream lease")
        return self


def _safe_root(path: Path) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise GenesisUiBundleError("Genesis demo root must be a regular non-symlink directory")
    return path.resolve(strict=True)


def _safe_path(root: Path, path: Path) -> Path:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise GenesisUiBundleError(f"artifact escapes the Genesis demo root: {path}") from exc
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise GenesisUiBundleError(f"artifact path contains a symlink: {current}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise GenesisUiBundleError(f"artifact is not a regular file: {path}")
    return resolved


def _read_bytes(root: Path, path: Path) -> bytes:
    resolved = _safe_path(root, path)
    size = resolved.stat().st_size
    if size > _MAX_DOCUMENT_BYTES:
        raise GenesisUiBundleError(f"artifact exceeds the bounded document size: {path}")
    try:
        return resolved.read_bytes()
    except OSError as exc:
        raise GenesisUiBundleError(f"cannot read Genesis artifact {path}: {exc}") from exc


def _decode_json(path: Path, payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            payload,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise GenesisUiBundleError(f"invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GenesisUiBundleError(f"Genesis artifact must be a JSON object: {path}")
    return value


def _read_json(root: Path, path: Path) -> dict[str, Any]:
    return _decode_json(path, _read_bytes(root, path))


def _artifact_json(
    root: Path,
    capsule_root: Path,
    capsule: GenesisCapsule,
    artifact_id: str,
) -> dict[str, Any]:
    artifact = next((item for item in capsule.artifacts if item.artifact_id == artifact_id), None)
    if artifact is None:
        raise GenesisUiBundleError(f"capsule omits required artifact {artifact_id}")
    path = capsule_root / artifact.path
    payload = _read_bytes(root, path)
    if len(payload) != artifact.size_bytes:
        raise GenesisUiBundleError(f"capsule artifact size mismatch: {artifact_id}")
    if hashlib.sha256(payload).hexdigest() != artifact.digest.value:
        raise GenesisUiBundleError(f"capsule artifact digest mismatch: {artifact_id}")
    return _decode_json(path, payload)


def _publish_once(bundle: GenesisUiBundle, output: Path) -> Path:
    if output.exists() or output.is_symlink():
        raise GenesisUiBundleError(f"refusing to overwrite Genesis UI bundle: {output}")
    if output.parent.exists() and output.parent.is_symlink():
        raise GenesisUiBundleError("Genesis UI bundle parent must not be a symlink")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(bundle) + b"\n"
    try:
        with output.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise GenesisUiBundleError(f"cannot publish Genesis UI bundle: {exc}") from exc
    return output.resolve(strict=True)


def export_genesis_ui_bundle(demo_root: Path, output: Path | None = None) -> Path:
    """Validate and combine exact flagship artifacts into one browser-loadable document."""

    root = _safe_root(demo_root)
    target = root / "genesis-ui-bundle.json" if output is None else output
    try:
        summary_path = root / "GENESIS_DEMO_REPORT.json"
        summary = DemoSummary.model_validate_json(_read_bytes(root, summary_path), strict=True)
        if Path(summary.output_directory).resolve() != root:
            raise GenesisUiBundleError("demo summary output directory differs from the bundle root")
        if Path(summary.report_path).resolve() != summary_path.resolve():
            raise GenesisUiBundleError("demo summary report path differs from its source artifact")
        if Path(summary.ui_bundle_path).resolve() != target.resolve():
            raise GenesisUiBundleError("demo summary UI bundle path differs from the export target")

        candidate_root = root / "run/candidates"
        if candidate_root.is_symlink() or not candidate_root.is_dir():
            raise GenesisUiBundleError("candidate artifact directory is missing or symlinked")
        candidates: list[CandidateBundle] = []
        for directory in sorted(candidate_root.iterdir(), key=lambda item: item.name):
            if directory.is_symlink() or not directory.is_dir():
                raise GenesisUiBundleError(
                    f"candidate entry is not a regular directory: {directory}"
                )
            candidate_path = directory / "candidate.json"
            candidate = load_candidate(_read_bytes(root, candidate_path))
            design = CandidateDesign.model_validate_json(
                _read_bytes(root, directory / "candidate_design.json"), strict=True
            )
            candidates.append(CandidateBundle(candidate=candidate, design=design))

        accepted_genome = load_inference_genome(
            _read_bytes(
                root,
                candidate_root / summary.accepted_candidate_id / "inference_genome.json",
            )
        )
        counterexample_root = root / "run/synthesis/cegis/counterexamples"
        counterexamples = tuple(
            load_counterexample(_read_bytes(root, path))
            for path in sorted(counterexample_root.glob("*.json"), key=lambda item: item.name)
        )

        capsule_manifest = root / f"capsule/manifests/{summary.capsule_digest}.json"
        if Path(summary.capsule_path).resolve() != capsule_manifest.resolve():
            raise GenesisUiBundleError("demo summary capsule path differs from its digest address")
        capsule = GenesisCapsule.model_validate_json(
            _read_bytes(root, capsule_manifest), strict=True
        )
        if (
            capsule.capsule_digest is None
            or calculate_capsule_digest(capsule) != capsule.capsule_digest
        ):
            raise GenesisUiBundleError("capsule manifest failed its content-addressed digest")
        if len(capsule.benchmarks) != 1:
            raise GenesisUiBundleError("Genesis UI bundle requires exactly one capsule benchmark")
        benchmark = capsule.benchmarks[0]
        capsule_root = root / "capsule"
        definition = _artifact_json(root, capsule_root, capsule, benchmark.definition_artifact_id)
        candidate_samples_raw = _artifact_json(
            root, capsule_root, capsule, benchmark.raw_samples_artifact_id
        )
        baseline_samples_raw = _artifact_json(
            root, capsule_root, capsule, benchmark.baseline_artifact_id
        )
        candidate_samples = RawBenchmarkSamples.model_validate_json(
            json.dumps(candidate_samples_raw, allow_nan=False), strict=True
        )
        baseline_samples = RawBenchmarkSamples.model_validate_json(
            json.dumps(baseline_samples_raw, allow_nan=False), strict=True
        )
        definition_ref = next(
            item
            for item in capsule.artifacts
            if item.artifact_id == benchmark.definition_artifact_id
        )
        for samples in (candidate_samples, baseline_samples):
            if samples.benchmark_definition_digest != definition_ref.digest:
                raise GenesisUiBundleError(
                    "raw samples do not reference the capsule benchmark definition"
                )

        evolution = EvolutionSnapshot.model_validate_json(
            _read_bytes(root, root / "evolution/degraded-snapshot.json"), strict=True
        )
        lineage = LineageTransferReport.model_validate_json(
            _read_bytes(root, root / "lineage/report.json"), strict=True
        )
        bundle = GenesisUiBundle(
            summary=summary,
            genome=accepted_genome,
            candidates=tuple(candidates),
            counterexamples=counterexamples,
            capsule=capsule,
            benchmark_definition=definition,
            baseline_samples=baseline_samples,
            candidate_samples=candidate_samples,
            evolution=evolution,
            lineage=lineage,
        )
    except (ValidationError, ValueError, OSError) as exc:
        if isinstance(exc, GenesisUiBundleError):
            raise
        raise GenesisUiBundleError(f"Genesis UI bundle validation failed: {exc}") from exc
    return _publish_once(bundle, target)


__all__ = [
    "GENESIS_UI_BUNDLE_VERSION",
    "GenesisUiBundle",
    "GenesisUiBundleError",
    "export_genesis_ui_bundle",
]
