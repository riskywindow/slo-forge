from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from sloforge.genesis.capsule import (
    ArtifactOrigin,
    ArtifactRef,
    ArtifactRole,
    BenchmarkEvidence,
    BenchmarkSummary,
    CapsuleIdentity,
    CapsuleIOError,
    ClaimCategory,
    ClaimScope,
    CounterexampleCorpus,
    CurrentDependency,
    DependencyRequirement,
    Digest,
    EvidenceClass,
    EvidenceIssuer,
    EvidenceRecord,
    EvidenceResult,
    GenesisCapsule,
    HardwareCompatibility,
    RawBenchmarkSample,
    RawBenchmarkSamples,
    ScopedClaim,
    ValidationContext,
    ValidationIssueCode,
    VerificationLevel,
    load_capsule,
    publish_capsule,
    seal_capsule,
    validate_capsule,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)
OBSERVED = datetime(2025, 12, 1, tzinfo=UTC)
VALID_UNTIL = datetime(2027, 1, 1, tzinfo=UTC)


def _digest(payload: bytes) -> Digest:
    return Digest(value=hashlib.sha256(payload).hexdigest())


def _constant_digest(label: str) -> Digest:
    return _digest(label.encode())


def _add_artifact(
    root: Path,
    artifact_id: str,
    role: ArtifactRole,
    payload: bytes,
    *,
    origin: ArtifactOrigin = ArtifactOrigin.VERIFIED_EVIDENCE,
) -> ArtifactRef:
    relative = f"artifacts/{artifact_id}"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return ArtifactRef(
        artifact_id=artifact_id,
        role=role,
        origin=origin,
        digest=_digest(payload),
        size_bytes=len(payload),
        path=relative,
        media_type="application/json",
    )


def _complete_capsule(root: Path) -> tuple[GenesisCapsule, ValidationContext]:
    candidate_hash = _constant_digest("candidate")
    hardware_hash = _constant_digest("hardware")
    definition = _add_artifact(root, "definition", ArtifactRole.BENCHMARK_DEFINITION, b"{}")
    software = _add_artifact(root, "software", ArtifactRole.SOFTWARE_MANIFEST, b"{}")
    lock = _add_artifact(
        root, "dependency-lock", ArtifactRole.DEPENDENCY_LOCK, b"runtime==1.0.0\n"
    )
    workload_hash = _constant_digest("workload")
    baseline_document = RawBenchmarkSamples(
        benchmark_definition_digest=definition.digest,
        workload_fingerprint=workload_hash,
        hardware_fingerprint=hardware_hash,
        software_manifest_digest=software.digest,
        samples=(
            RawBenchmarkSample(trial=0, seed=21, value=12.0),
            RawBenchmarkSample(trial=1, seed=22, value=14.0),
        ),
    )
    baseline = _add_artifact(
        root,
        "baseline",
        ArtifactRole.PERFORMANCE_SAMPLES,
        baseline_document.model_dump_json().encode(),
        origin=ArtifactOrigin.PERFORMANCE_EVIDENCE,
    )
    samples_document = RawBenchmarkSamples(
        benchmark_definition_digest=definition.digest,
        workload_fingerprint=workload_hash,
        hardware_fingerprint=hardware_hash,
        software_manifest_digest=software.digest,
        samples=(
            RawBenchmarkSample(trial=0, seed=11, value=9.0),
            RawBenchmarkSample(trial=1, seed=12, value=11.0),
        ),
    )
    samples = _add_artifact(
        root,
        "samples",
        ArtifactRole.PERFORMANCE_SAMPLES,
        samples_document.model_dump_json().encode(),
        origin=ArtifactOrigin.PERFORMANCE_EVIDENCE,
    )
    runtime = _add_artifact(
        root,
        "runtime",
        ArtifactRole.GENERATED_RUNTIME,
        b"generated runtime",
        origin=ArtifactOrigin.GENERATED_UNTRUSTED,
    )
    deployment = _add_artifact(root, "deployment", ArtifactRole.DEPLOYMENT, b"deployment")
    rollback = _add_artifact(root, "rollback", ArtifactRole.ROLLBACK, b"rollback")
    evidence_artifacts: dict[EvidenceClass, ArtifactRef] = {}
    role_by_class = {
        EvidenceClass.SEMANTIC: ArtifactRole.SEMANTIC_EVIDENCE,
        EvidenceClass.QUALITY: ArtifactRole.QUALITY_EVIDENCE,
        EvidenceClass.RESOURCE: ArtifactRole.RESOURCE_EVIDENCE,
        EvidenceClass.PERFORMANCE: ArtifactRole.PERFORMANCE_SAMPLES,
        EvidenceClass.OPERATIONAL: ArtifactRole.OPERATIONAL_EVIDENCE,
    }
    for evidence_class, role in role_by_class.items():
        evidence_artifacts[evidence_class] = (
            samples
            if evidence_class is EvidenceClass.PERFORMANCE
            else _add_artifact(root, f"evidence-{evidence_class.value}", role, b"{}")
        )
    corpus_document = CounterexampleCorpus(
        candidate_genome_hash=candidate_hash,
        counterexample_artifact_ids=(),
        searched_domains=("supported shape and protocol domain",),
    )
    corpus = _add_artifact(
        root,
        "counterexamples",
        ArtifactRole.COUNTEREXAMPLE_CORPUS,
        corpus_document.model_dump_json().encode(),
    )
    issuer_by_class = {
        EvidenceClass.SEMANTIC: EvidenceIssuer.OPERATOR_VERIFIER,
        EvidenceClass.QUALITY: EvidenceIssuer.QUALITY_HARNESS,
        EvidenceClass.RESOURCE: EvidenceIssuer.RESOURCE_ANALYZER,
        EvidenceClass.PERFORMANCE: EvidenceIssuer.BENCHMARK_HARNESS,
        EvidenceClass.OPERATIONAL: EvidenceIssuer.MODEL_CHECKER,
    }
    evidence = tuple(
        EvidenceRecord(
            evidence_id=f"evidence:{evidence_class.value}",
            evidence_class=evidence_class,
            level=(
                VerificationLevel.HARDWARE_OPERATIONAL
                if evidence_class in {EvidenceClass.PERFORMANCE, EvidenceClass.OPERATIONAL}
                else VerificationLevel.PROPERTY
            ),
            result=EvidenceResult.PASS,
            issuer=issuer_by_class[evidence_class],
            issuer_version="1.0.0",
            artifact_ids=(artifact.artifact_id,),
            observed_at=OBSERVED,
            valid_until=VALID_UNTIL,
            deterministic_seed=73129,
        )
        for evidence_class, artifact in evidence_artifacts.items()
    )
    category_class = {
        ClaimCategory.SEMANTIC: EvidenceClass.SEMANTIC,
        ClaimCategory.QUALITY: EvidenceClass.QUALITY,
        ClaimCategory.RESOURCE: EvidenceClass.RESOURCE,
        ClaimCategory.PERFORMANCE: EvidenceClass.PERFORMANCE,
        ClaimCategory.OPERATIONAL: EvidenceClass.OPERATIONAL,
    }
    claims = tuple(
        ScopedClaim(
            claim_id=f"claim:{category.value}",
            category=category,
            statement=f"scoped {category.value} claim",
            scope=ClaimScope(
                input_domain=("tokens length 1..32",),
                hardware_fingerprints=(hardware_hash,),
            ),
            level=(
                VerificationLevel.HARDWARE_OPERATIONAL
                if category in {ClaimCategory.PERFORMANCE, ClaimCategory.OPERATIONAL}
                else VerificationLevel.PROPERTY
            ),
            result=EvidenceResult.PASS,
            evidence_ids=(f"evidence:{evidence_class.value}",),
        )
        for category, evidence_class in category_class.items()
    )
    hardware = HardwareCompatibility(
        hardware_contract_hash=hardware_hash,
        allowed_fingerprints=(hardware_hash,),
        architectures=("cpu-test",),
    )
    capsule = GenesisCapsule(
        identity=CapsuleIdentity(
            candidate_genome_hash=candidate_hash,
            source_model_hash=_constant_digest("model"),
            tokenizer_hash=_constant_digest("tokenizer"),
            workload_contract_hash=workload_hash,
            hardware_contract_hash=hardware_hash,
            compiler_version="1.0.0",
            verifier_version="1.0.0",
            git_commit="abcdef0",
            dependency_lock_hash=lock.digest,
            generated_at=OBSERVED,
        ),
        artifacts=(
            definition,
            software,
            baseline,
            lock,
            samples,
            runtime,
            deployment,
            rollback,
            *(
                artifact
                for evidence_class, artifact in evidence_artifacts.items()
                if evidence_class is not EvidenceClass.PERFORMANCE
            ),
            corpus,
        ),
        dependencies=(DependencyRequirement(name="runtime", version="1.0.0"),),
        hardware=hardware,
        evidence=evidence,
        claims=claims,
        benchmarks=(
            BenchmarkEvidence(
                benchmark_id="end-to-end",
                definition_artifact_id=definition.artifact_id,
                raw_samples_artifact_id=samples.artifact_id,
                software_manifest_artifact_id=software.artifact_id,
                baseline_artifact_id=baseline.artifact_id,
                workload_fingerprint=workload_hash,
                hardware_fingerprint=hardware_hash,
                sample_count=2,
                warmup_iterations=2,
                repetitions=2,
                randomized_run_order=True,
                noise_floor=0.5,
                summary=BenchmarkSummary(
                    metric="latency",
                    unit="milliseconds",
                    objective="minimize",
                    tail_quantile=0.9,
                    median=10.0,
                    tail_percentile=10.8,
                    confidence_low=9.0,
                    confidence_high=11.0,
                    effect_size=3.0 / 13.0,
                    regression_probability=0.01,
                    practical_significance_threshold=0.1,
                ),
            ),
        ),
    )
    context = ValidationContext(
        source_model_hash=_constant_digest("model"),
        tokenizer_hash=_constant_digest("tokenizer"),
        workload_contract_hash=workload_hash,
        hardware_contract_hash=hardware_hash,
        hardware_fingerprint=hardware_hash,
        hardware_architecture="cpu-test",
        device_count=1,
        dependency_lock_hash=lock.digest,
        dependencies=(CurrentDependency(name="runtime", version="1.0.0"),),
        trusted_verifier_version="1.0.0",
        now=NOW,
    )
    return seal_capsule(capsule), context


def _codes(capsule: GenesisCapsule, root: Path, context: ValidationContext) -> set[str]:
    return {issue.code.value for issue in validate_capsule(capsule, root, context).issues}


def test_complete_capsule_is_promotion_eligible(tmp_path: Path) -> None:
    capsule, context = _complete_capsule(tmp_path)
    report = validate_capsule(capsule, tmp_path, context)
    assert report.integrity_valid
    assert report.contract_compatible
    assert report.evidence_complete
    assert report.promotion_eligible
    assert report.issues == ()


def test_manifest_and_artifact_tampering_are_detected(tmp_path: Path) -> None:
    capsule, context = _complete_capsule(tmp_path)
    tampered_manifest = capsule.model_copy(update={"known_unsupported_cases": ("new",)})
    assert ValidationIssueCode.MANIFEST_TAMPERED.value in _codes(
        tampered_manifest, tmp_path, context
    )

    runtime = next(item for item in capsule.artifacts if item.artifact_id == "runtime")
    (tmp_path / runtime.path).write_bytes(b"tampered")
    codes = _codes(capsule, tmp_path, context)
    assert ValidationIssueCode.ARTIFACT_TAMPERED.value in codes
    assert ValidationIssueCode.ARTIFACT_SIZE_MISMATCH.value in codes


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("stale", ValidationIssueCode.EVIDENCE_STALE),
        ("hardware", ValidationIssueCode.HARDWARE_MISMATCH),
        ("dependency", ValidationIssueCode.DEPENDENCY_MISMATCH),
        ("contract", ValidationIssueCode.CONTRACT_MISMATCH),
        ("verifier", ValidationIssueCode.VERIFIER_MISMATCH),
    ],
)
def test_stale_or_incompatible_capsules_are_rejected(
    tmp_path: Path, mutation: str, expected: ValidationIssueCode
) -> None:
    capsule, context = _complete_capsule(tmp_path)
    if mutation == "stale":
        context = context.model_copy(update={"now": datetime(2030, 1, 1, tzinfo=UTC)})
    elif mutation == "hardware":
        context = context.model_copy(update={"hardware_fingerprint": _constant_digest("other")})
    elif mutation == "dependency":
        context = context.model_copy(
            update={"dependencies": (CurrentDependency(name="runtime", version="2.0.0"),)}
        )
    elif mutation == "contract":
        context = context.model_copy(update={"source_model_hash": _constant_digest("other")})
    else:
        context = context.model_copy(update={"trusted_verifier_version": "2.0.0"})
    report = validate_capsule(capsule, tmp_path, context)
    assert not report.promotion_eligible
    assert expected.value in {issue.code.value for issue in report.issues}


def test_missing_counterexample_or_evidence_rejects_promotion(tmp_path: Path) -> None:
    capsule, context = _complete_capsule(tmp_path)
    no_corpus = seal_capsule(
        capsule.model_copy(
            update={
                "capsule_digest": None,
                "artifacts": tuple(
                    item
                    for item in capsule.artifacts
                    if item.role is not ArtifactRole.COUNTEREXAMPLE_CORPUS
                ),
            }
        )
    )
    assert ValidationIssueCode.COUNTEREXAMPLE_CORPUS_MISSING.value in _codes(
        no_corpus, tmp_path, context
    )

    incomplete = seal_capsule(
        capsule.model_copy(
            update={
                "capsule_digest": None,
                "evidence": tuple(
                    item
                    for item in capsule.evidence
                    if item.evidence_class is not EvidenceClass.QUALITY
                ),
            }
        )
    )
    codes = _codes(incomplete, tmp_path, context)
    assert ValidationIssueCode.REQUIRED_EVIDENCE_CLASS_MISSING.value in codes
    assert ValidationIssueCode.EVIDENCE_INCOMPLETE.value in codes


def test_resealed_benchmark_with_altered_result_is_rejected(tmp_path: Path) -> None:
    capsule, context = _complete_capsule(tmp_path)
    benchmark = capsule.benchmarks[0]
    altered = benchmark.model_copy(
        update={"summary": benchmark.summary.model_copy(update={"median": 9.5})}
    )
    resealed = seal_capsule(
        capsule.model_copy(update={"capsule_digest": None, "benchmarks": (altered,)})
    )
    assert ValidationIssueCode.BENCHMARK_PROVENANCE_INVALID.value in _codes(
        resealed, tmp_path, context
    )


def test_symlinked_capsule_artifact_is_rejected(tmp_path: Path) -> None:
    capsule, context = _complete_capsule(tmp_path)
    runtime = next(item for item in capsule.artifacts if item.artifact_id == "runtime")
    runtime_path = tmp_path / runtime.path
    outside = tmp_path / "outside"
    outside.write_bytes(runtime_path.read_bytes())
    runtime_path.unlink()
    runtime_path.symlink_to(outside)
    assert ValidationIssueCode.UNSAFE_ARTIFACT_PATH.value in _codes(capsule, tmp_path, context)


def test_capsule_json_is_strict(tmp_path: Path) -> None:
    capsule, _ = _complete_capsule(tmp_path)
    document = json.loads(capsule.model_dump_json())
    document["unexpected"] = True
    with pytest.raises(ValueError):
        GenesisCapsule.model_validate(document, strict=True)


def test_capsule_matches_checked_in_json_schema(tmp_path: Path) -> None:
    capsule, _ = _complete_capsule(tmp_path)
    schema_path = Path("schemas/genesis_capsule/genesis-capsule-v1.schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(capsule.model_dump(mode="json"))


def test_capsule_publication_is_content_addressed_and_immutable(tmp_path: Path) -> None:
    capsule, _ = _complete_capsule(tmp_path / "bundle")
    manifest = publish_capsule(capsule, tmp_path / "published")
    assert manifest.name == f"{capsule.capsule_digest.value}.json"
    assert load_capsule(manifest) == capsule
    assert publish_capsule(capsule, tmp_path / "published") == manifest

    manifest.chmod(0o644)
    manifest.write_text("{}\n", encoding="utf-8")
    with pytest.raises(CapsuleIOError, match="immutable-content"):
        publish_capsule(capsule, tmp_path / "published")
