"""Independent, fail-closed Genesis capsule validator."""

from __future__ import annotations

import hashlib
import math
import statistics
from pathlib import Path

from pydantic import ValidationError

from .canonical import calculate_capsule_digest
from .models import (
    ArtifactRef,
    ArtifactRole,
    BenchmarkEvidence,
    CapsuleValidationReport,
    ClaimCategory,
    CounterexampleCorpus,
    EvidenceClass,
    EvidenceIssuer,
    EvidenceResult,
    GenesisCapsule,
    RawBenchmarkSamples,
    ValidationContext,
    ValidationIssue,
    ValidationIssueCode,
    verification_level_rank,
)

_PROMOTION_ARTIFACT_ROLES = frozenset(
    {ArtifactRole.GENERATED_RUNTIME, ArtifactRole.DEPLOYMENT, ArtifactRole.ROLLBACK}
)
_PROMOTION_EVIDENCE_CLASSES = frozenset(
    {
        EvidenceClass.SEMANTIC,
        EvidenceClass.QUALITY,
        EvidenceClass.RESOURCE,
        EvidenceClass.PERFORMANCE,
        EvidenceClass.OPERATIONAL,
    }
)
_PROMOTION_CLAIM_CATEGORIES = frozenset(
    {
        ClaimCategory.SEMANTIC,
        ClaimCategory.QUALITY,
        ClaimCategory.RESOURCE,
        ClaimCategory.PERFORMANCE,
        ClaimCategory.OPERATIONAL,
    }
)
_ISSUERS_BY_CLASS = {
    EvidenceClass.BUILD: frozenset({EvidenceIssuer.TRUSTED_VALIDATOR, EvidenceIssuer.SANDBOX}),
    EvidenceClass.SEMANTIC: frozenset(
        {EvidenceIssuer.TRUSTED_VALIDATOR, EvidenceIssuer.OPERATOR_VERIFIER}
    ),
    EvidenceClass.QUALITY: frozenset({EvidenceIssuer.QUALITY_HARNESS}),
    EvidenceClass.RESOURCE: frozenset({EvidenceIssuer.RESOURCE_ANALYZER}),
    EvidenceClass.PERFORMANCE: frozenset({EvidenceIssuer.BENCHMARK_HARNESS}),
    EvidenceClass.OPERATIONAL: frozenset(
        {EvidenceIssuer.TRUSTED_VALIDATOR, EvidenceIssuer.MODEL_CHECKER}
    ),
    EvidenceClass.MODEL_CHECK: frozenset({EvidenceIssuer.MODEL_CHECKER}),
    EvidenceClass.PROPERTY_TEST: frozenset({EvidenceIssuer.PROPERTY_HARNESS}),
    EvidenceClass.FUZZ: frozenset({EvidenceIssuer.FUZZ_HARNESS}),
    EvidenceClass.DIFFERENTIAL: frozenset({EvidenceIssuer.OPERATOR_VERIFIER}),
}
_EVIDENCE_CLASS_BY_CLAIM = {
    ClaimCategory.BUILD: EvidenceClass.BUILD,
    ClaimCategory.SEMANTIC: EvidenceClass.SEMANTIC,
    ClaimCategory.QUALITY: EvidenceClass.QUALITY,
    ClaimCategory.RESOURCE: EvidenceClass.RESOURCE,
    ClaimCategory.PERFORMANCE: EvidenceClass.PERFORMANCE,
    ClaimCategory.OPERATIONAL: EvidenceClass.OPERATIONAL,
}
_ARTIFACT_ROLE_BY_EVIDENCE = {
    EvidenceClass.BUILD: frozenset(
        {ArtifactRole.COMPILED_BINARY, ArtifactRole.GENERATED_RUNTIME}
    ),
    EvidenceClass.SEMANTIC: frozenset(
        {ArtifactRole.SEMANTIC_EVIDENCE, ArtifactRole.DIFFERENTIAL_TEST_RESULT}
    ),
    EvidenceClass.QUALITY: frozenset({ArtifactRole.QUALITY_EVIDENCE}),
    EvidenceClass.RESOURCE: frozenset({ArtifactRole.RESOURCE_EVIDENCE}),
    EvidenceClass.PERFORMANCE: frozenset({ArtifactRole.PERFORMANCE_SAMPLES}),
    EvidenceClass.OPERATIONAL: frozenset(
        {ArtifactRole.OPERATIONAL_EVIDENCE, ArtifactRole.MODEL_CHECK_RESULT}
    ),
    EvidenceClass.MODEL_CHECK: frozenset({ArtifactRole.MODEL_CHECK_RESULT}),
    EvidenceClass.PROPERTY_TEST: frozenset({ArtifactRole.PROPERTY_TEST_RESULT}),
    EvidenceClass.FUZZ: frozenset({ArtifactRole.FUZZ_RESULT}),
    EvidenceClass.DIFFERENTIAL: frozenset({ArtifactRole.DIFFERENTIAL_TEST_RESULT}),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_artifact(root: Path, artifact: ArtifactRef) -> Path | None:
    """Resolve an artifact without allowing symlinks to escape the capsule."""

    candidate = root.joinpath(*artifact.path.split("/"))
    try:
        if candidate.is_symlink():
            return None
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (FileNotFoundError, OSError, ValueError):
        return None
    if not resolved.is_file():
        return None
    return resolved


def _append(
    issues: list[ValidationIssue], code: ValidationIssueCode, path: str, message: str
) -> None:
    issues.append(ValidationIssue(code=code, path=path, message=message))


def _validate_benchmark(
    benchmark: BenchmarkEvidence,
    artifacts: dict[str, ArtifactRef],
    resolved: dict[str, Path],
    context: ValidationContext,
    issues: list[ValidationIssue],
) -> None:
    prefix = f"benchmarks.{benchmark.benchmark_id}"
    required_ids = {
        benchmark.definition_artifact_id,
        benchmark.raw_samples_artifact_id,
        benchmark.software_manifest_artifact_id,
        benchmark.baseline_artifact_id,
    }
    if not required_ids.issubset(artifacts):
        _append(
            issues,
            ValidationIssueCode.BENCHMARK_PROVENANCE_INVALID,
            prefix,
            "benchmark references artifacts absent from the manifest",
        )
        return
    required_roles = {
        benchmark.definition_artifact_id: ArtifactRole.BENCHMARK_DEFINITION,
        benchmark.raw_samples_artifact_id: ArtifactRole.PERFORMANCE_SAMPLES,
        benchmark.software_manifest_artifact_id: ArtifactRole.SOFTWARE_MANIFEST,
        benchmark.baseline_artifact_id: ArtifactRole.PERFORMANCE_SAMPLES,
    }
    if any(artifacts[item].role is not role for item, role in required_roles.items()):
        _append(
            issues,
            ValidationIssueCode.BENCHMARK_PROVENANCE_INVALID,
            prefix,
            "benchmark artifact roles do not match their declared provenance fields",
        )
    raw_path = resolved.get(benchmark.raw_samples_artifact_id)
    baseline_path = resolved.get(benchmark.baseline_artifact_id)
    if raw_path is None or baseline_path is None:
        return
    try:
        samples = RawBenchmarkSamples.model_validate_json(raw_path.read_bytes(), strict=True)
        baseline_samples = RawBenchmarkSamples.model_validate_json(
            baseline_path.read_bytes(), strict=True
        )
    except (OSError, ValidationError) as exc:
        _append(
            issues,
            ValidationIssueCode.BENCHMARK_PROVENANCE_INVALID,
            f"{prefix}.raw_samples",
            f"raw samples are not a valid evidence document: {exc}",
        )
        return
    definition = artifacts[benchmark.definition_artifact_id]
    software = artifacts[benchmark.software_manifest_artifact_id]
    mismatches: list[str] = []
    if len(samples.samples) != benchmark.sample_count:
        mismatches.append("sample count")
    if benchmark.repetitions != benchmark.sample_count:
        mismatches.append("repetition count")
    if samples.benchmark_definition_digest != definition.digest:
        mismatches.append("benchmark definition digest")
    if samples.software_manifest_digest != software.digest:
        mismatches.append("software manifest digest")
    if samples.workload_fingerprint != benchmark.workload_fingerprint:
        mismatches.append("workload fingerprint")
    if samples.hardware_fingerprint != benchmark.hardware_fingerprint:
        mismatches.append("hardware fingerprint")
    if baseline_samples.benchmark_definition_digest != definition.digest:
        mismatches.append("baseline benchmark definition digest")
    if baseline_samples.software_manifest_digest != software.digest:
        mismatches.append("baseline software manifest digest")
    if baseline_samples.workload_fingerprint != benchmark.workload_fingerprint:
        mismatches.append("baseline workload fingerprint")
    if baseline_samples.hardware_fingerprint != benchmark.hardware_fingerprint:
        mismatches.append("baseline hardware fingerprint")
    if benchmark.hardware_fingerprint != context.hardware_fingerprint:
        mismatches.append("validation hardware fingerprint")
    if not benchmark.randomized_run_order:
        mismatches.append("randomized run order")
    values = sorted(sample.value for sample in samples.samples)
    median = statistics.median(values)
    baseline_median = statistics.median(sample.value for sample in baseline_samples.samples)
    tail_position = (len(values) - 1) * benchmark.summary.tail_quantile
    tail_lower = math.floor(tail_position)
    tail_upper = math.ceil(tail_position)
    tail_fraction = tail_position - tail_lower
    tail = values[tail_lower] * (1.0 - tail_fraction) + values[tail_upper] * tail_fraction
    if not math.isclose(median, benchmark.summary.median, rel_tol=1e-12, abs_tol=1e-12):
        mismatches.append("reported median")
    if not math.isclose(tail, benchmark.summary.tail_percentile, rel_tol=1e-12, abs_tol=1e-12):
        mismatches.append("reported tail percentile")
    if baseline_median == 0.0:
        expected_effect = 0.0 if median == 0.0 else math.inf
    elif benchmark.summary.objective == "minimize":
        expected_effect = (baseline_median - median) / abs(baseline_median)
    else:
        expected_effect = (median - baseline_median) / abs(baseline_median)
    if not math.isfinite(expected_effect) or not math.isclose(
        expected_effect, benchmark.summary.effect_size, rel_tol=1e-12, abs_tol=1e-12
    ):
        mismatches.append("reported effect size")
    if mismatches:
        _append(
            issues,
            ValidationIssueCode.BENCHMARK_PROVENANCE_INVALID,
            prefix,
            "invalid " + ", ".join(mismatches),
        )


def validate_capsule(
    capsule: GenesisCapsule, capsule_root: Path, context: ValidationContext
) -> CapsuleValidationReport:
    """Validate manifest integrity, scoped evidence, and promotion compatibility.

    No compiler or generated-runtime code is imported or executed here.
    Every failure is accumulated into a deterministic report.
    """

    issues: list[ValidationIssue] = []
    if capsule.capsule_digest is None:
        _append(
            issues,
            ValidationIssueCode.UNSEALED,
            "capsule_digest",
            "capsule has not been content-addressed",
        )
    elif calculate_capsule_digest(capsule) != capsule.capsule_digest:
        _append(
            issues,
            ValidationIssueCode.MANIFEST_TAMPERED,
            "capsule_digest",
            "manifest content does not match its declared digest",
        )

    artifacts = {artifact.artifact_id: artifact for artifact in capsule.artifacts}
    resolved: dict[str, Path] = {}
    for artifact in capsule.artifacts:
        path = _resolve_artifact(capsule_root, artifact)
        if path is None:
            candidate = capsule_root.joinpath(*artifact.path.split("/"))
            code = (
                ValidationIssueCode.ARTIFACT_MISSING
                if not candidate.exists()
                else ValidationIssueCode.UNSAFE_ARTIFACT_PATH
            )
            _append(issues, code, f"artifacts.{artifact.artifact_id}", "artifact is unavailable")
            continue
        resolved[artifact.artifact_id] = path
        actual_size = path.stat().st_size
        if actual_size != artifact.size_bytes:
            _append(
                issues,
                ValidationIssueCode.ARTIFACT_SIZE_MISMATCH,
                f"artifacts.{artifact.artifact_id}",
                f"declared {artifact.size_bytes} bytes but found {actual_size}",
            )
        if _sha256_file(path) != artifact.digest.value:
            _append(
                issues,
                ValidationIssueCode.ARTIFACT_TAMPERED,
                f"artifacts.{artifact.artifact_id}",
                "artifact content does not match its declared digest",
            )

    evidence = {record.evidence_id: record for record in capsule.evidence}
    for record in capsule.evidence:
        prefix = f"evidence.{record.evidence_id}"
        if any(artifact_id not in artifacts for artifact_id in record.artifact_ids):
            _append(
                issues,
                ValidationIssueCode.EVIDENCE_INCOMPLETE,
                prefix,
                "evidence references an artifact absent from the manifest",
            )
        if any(artifact_id not in resolved for artifact_id in record.artifact_ids):
            _append(
                issues,
                ValidationIssueCode.EVIDENCE_INCOMPLETE,
                prefix,
                "evidence has unavailable or invalid artifact content",
            )
        if record.result is not EvidenceResult.PASS:
            _append(
                issues,
                ValidationIssueCode.EVIDENCE_FAILED,
                prefix,
                f"evidence result is {record.result.value}",
            )
        if record.valid_until is None or record.valid_until <= context.now:
            _append(
                issues,
                ValidationIssueCode.EVIDENCE_STALE,
                prefix,
                "evidence has expired or has no validity horizon",
            )
        if record.observed_at > context.now:
            _append(
                issues,
                ValidationIssueCode.EVIDENCE_STALE,
                prefix,
                "evidence observation time is in the future",
            )
        if record.observed_at > capsule.identity.generated_at:
            _append(
                issues,
                ValidationIssueCode.EVIDENCE_STALE,
                prefix,
                "evidence was observed after the capsule generation time",
            )
        if record.issuer not in _ISSUERS_BY_CLASS[record.evidence_class]:
            _append(
                issues,
                ValidationIssueCode.EVIDENCE_INCOMPLETE,
                prefix,
                "evidence class was not produced by an allowed independent issuer",
            )
        referenced_roles = {
            artifacts[item].role for item in record.artifact_ids if item in artifacts
        }
        if referenced_roles and not referenced_roles.intersection(
            _ARTIFACT_ROLE_BY_EVIDENCE[record.evidence_class]
        ):
            _append(
                issues,
                ValidationIssueCode.EVIDENCE_INCOMPLETE,
                prefix,
                "evidence does not reference an artifact with a compatible evidence role",
            )

    for claim in capsule.claims:
        prefix = f"claims.{claim.claim_id}"
        records = [evidence[item] for item in claim.evidence_ids if item in evidence]
        if len(records) != len(claim.evidence_ids):
            _append(
                issues,
                ValidationIssueCode.EVIDENCE_INCOMPLETE,
                prefix,
                "claim references evidence absent from the manifest",
            )
        if claim.result is not EvidenceResult.PASS:
            _append(
                issues,
                ValidationIssueCode.EVIDENCE_FAILED,
                prefix,
                f"claim result is {claim.result.value}",
            )
        if records and max(verification_level_rank(item.level) for item in records) < (
            verification_level_rank(claim.level)
        ):
            _append(
                issues,
                ValidationIssueCode.EVIDENCE_LEVEL_MISMATCH,
                prefix,
                "claim level exceeds every referenced evidence level",
            )
        if records and not any(
            item.evidence_class is _EVIDENCE_CLASS_BY_CLAIM[claim.category] for item in records
        ):
            _append(
                issues,
                ValidationIssueCode.EVIDENCE_INCOMPLETE,
                prefix,
                "claim category is not supported by matching evidence",
            )
        if claim.scope.hardware_fingerprints and context.hardware_fingerprint not in (
            claim.scope.hardware_fingerprints
        ):
            _append(
                issues,
                ValidationIssueCode.CLAIM_SCOPE_MISMATCH,
                prefix,
                "current hardware is outside the declared claim scope",
            )

    if capsule.identity.generated_at > context.now:
        _append(
            issues,
            ValidationIssueCode.EVIDENCE_STALE,
            "identity.generated_at",
            "capsule generation time is in the future",
        )
    identity_contracts = (
        ("source_model_hash", capsule.identity.source_model_hash, context.source_model_hash),
        ("tokenizer_hash", capsule.identity.tokenizer_hash, context.tokenizer_hash),
        (
            "workload_contract_hash",
            capsule.identity.workload_contract_hash,
            context.workload_contract_hash,
        ),
        (
            "hardware_contract_hash",
            capsule.identity.hardware_contract_hash,
            context.hardware_contract_hash,
        ),
    )
    for name, declared, current_contract in identity_contracts:
        if declared != current_contract:
            _append(
                issues,
                ValidationIssueCode.CONTRACT_MISMATCH,
                f"identity.{name}",
                "current contract hash differs from the capsule scope",
            )
    if capsule.identity.verifier_version != context.trusted_verifier_version:
        _append(
            issues,
            ValidationIssueCode.VERIFIER_MISMATCH,
            "identity.verifier_version",
            "capsule evidence was produced for a different verifier version",
        )
    if capsule.identity.dependency_lock_hash != context.dependency_lock_hash:
        _append(
            issues,
            ValidationIssueCode.DEPENDENCY_MISMATCH,
            "identity.dependency_lock_hash",
            "current dependency lock differs from the capsule lock",
        )
    lock_artifacts = [
        item for item in capsule.artifacts if item.role is ArtifactRole.DEPENDENCY_LOCK
    ]
    if len(lock_artifacts) != 1 or (
        lock_artifacts and lock_artifacts[0].digest != capsule.identity.dependency_lock_hash
    ):
        _append(
            issues,
            ValidationIssueCode.DEPENDENCY_MISMATCH,
            "artifacts",
            "capsule must contain the exact dependency lock named by its identity",
        )

    if context.hardware_fingerprint not in capsule.hardware.allowed_fingerprints:
        _append(
            issues,
            ValidationIssueCode.HARDWARE_MISMATCH,
            "hardware.allowed_fingerprints",
            "current hardware fingerprint is not capsule-compatible",
        )
    if context.hardware_architecture not in capsule.hardware.architectures:
        _append(
            issues,
            ValidationIssueCode.HARDWARE_MISMATCH,
            "hardware.architectures",
            "current hardware architecture is not capsule-compatible",
        )
    if context.device_count < capsule.hardware.minimum_device_count:
        _append(
            issues,
            ValidationIssueCode.HARDWARE_MISMATCH,
            "hardware.minimum_device_count",
            "current device count is below the capsule minimum",
        )

    current_dependencies = {item.name: item for item in context.dependencies}
    for required in capsule.dependencies:
        current_dependency = current_dependencies.get(required.name)
        prefix = f"dependencies.{required.name}"
        if current_dependency is None:
            _append(
                issues,
                ValidationIssueCode.DEPENDENCY_MISSING,
                prefix,
                "required dependency is absent",
            )
            continue
        if current_dependency.version != required.version or (
            required.package_digest is not None
            and current_dependency.package_digest != required.package_digest
        ):
            _append(
                issues,
                ValidationIssueCode.DEPENDENCY_MISMATCH,
                prefix,
                "installed dependency version or package digest differs",
            )

    for benchmark in capsule.benchmarks:
        _validate_benchmark(benchmark, artifacts, resolved, context, issues)

    if context.require_promotion_evidence:
        roles = {artifact.role for artifact in capsule.artifacts}
        for role in sorted(_PROMOTION_ARTIFACT_ROLES - roles, key=lambda item: item.value):
            _append(
                issues,
                ValidationIssueCode.REQUIRED_ARTIFACT_MISSING,
                "artifacts",
                f"promotion requires artifact role {role.value}",
            )
        classes = {record.evidence_class for record in capsule.evidence}
        for evidence_class in sorted(
            _PROMOTION_EVIDENCE_CLASSES - classes, key=lambda item: item.value
        ):
            _append(
                issues,
                ValidationIssueCode.REQUIRED_EVIDENCE_CLASS_MISSING,
                "evidence",
                f"promotion requires {evidence_class.value} evidence",
            )
        claim_categories = {claim.category for claim in capsule.claims if claim.promotion_required}
        for category in sorted(
            _PROMOTION_CLAIM_CATEGORIES - claim_categories, key=lambda item: item.value
        ):
            _append(
                issues,
                ValidationIssueCode.EVIDENCE_INCOMPLETE,
                "claims",
                f"promotion requires a scoped {category.value} claim",
            )
        corpus_refs = [
            item for item in capsule.artifacts if item.role is ArtifactRole.COUNTEREXAMPLE_CORPUS
        ]
        if len(corpus_refs) != 1:
            _append(
                issues,
                ValidationIssueCode.COUNTEREXAMPLE_CORPUS_MISSING,
                "artifacts",
                "promotion requires exactly one counterexample corpus",
            )
        elif corpus_refs[0].artifact_id in resolved:
            try:
                corpus = CounterexampleCorpus.model_validate_json(
                    resolved[corpus_refs[0].artifact_id].read_bytes(), strict=True
                )
                if corpus.candidate_genome_hash != capsule.identity.candidate_genome_hash:
                    raise ValueError("counterexample corpus belongs to another candidate")
                if set(corpus.counterexample_artifact_ids) - artifacts.keys():
                    raise ValueError("counterexample corpus references absent artifacts")
            except (OSError, ValidationError, ValueError) as exc:
                _append(
                    issues,
                    ValidationIssueCode.COUNTEREXAMPLE_CORPUS_MISSING,
                    f"artifacts.{corpus_refs[0].artifact_id}",
                    f"counterexample corpus is invalid: {exc}",
                )
        if not capsule.benchmarks:
            _append(
                issues,
                ValidationIssueCode.BENCHMARK_PROVENANCE_INVALID,
                "benchmarks",
                "promotion requires independent raw benchmark evidence",
            )
        else:
            raw_sample_ids = {item.raw_samples_artifact_id for item in capsule.benchmarks}
            performance_records = [
                item
                for item in capsule.evidence
                if item.evidence_class is EvidenceClass.PERFORMANCE
            ]
            if not any(
                raw_sample_ids.intersection(item.artifact_ids) for item in performance_records
            ):
                _append(
                    issues,
                    ValidationIssueCode.BENCHMARK_PROVENANCE_INVALID,
                    "evidence",
                    "performance claim evidence is not bound to benchmark raw samples",
                )

    integrity_codes = {
        ValidationIssueCode.UNSEALED,
        ValidationIssueCode.MANIFEST_TAMPERED,
        ValidationIssueCode.ARTIFACT_MISSING,
        ValidationIssueCode.ARTIFACT_TAMPERED,
        ValidationIssueCode.ARTIFACT_SIZE_MISMATCH,
        ValidationIssueCode.UNSAFE_ARTIFACT_PATH,
    }
    compatibility_codes = {
        ValidationIssueCode.CONTRACT_MISMATCH,
        ValidationIssueCode.HARDWARE_MISMATCH,
        ValidationIssueCode.DEPENDENCY_MISSING,
        ValidationIssueCode.DEPENDENCY_MISMATCH,
        ValidationIssueCode.CLAIM_SCOPE_MISMATCH,
        ValidationIssueCode.VERIFIER_MISMATCH,
    }
    integrity_valid = not any(issue.code in integrity_codes for issue in issues)
    contract_compatible = not any(issue.code in compatibility_codes for issue in issues)
    evidence_complete = not any(
        issue.code not in integrity_codes | compatibility_codes for issue in issues
    )
    return CapsuleValidationReport(
        capsule_digest=capsule.capsule_digest,
        integrity_valid=integrity_valid,
        contract_compatible=contract_compatible,
        evidence_complete=evidence_complete,
        promotion_eligible=not issues,
        checked_at=context.now,
        issues=tuple(issues),
    )
