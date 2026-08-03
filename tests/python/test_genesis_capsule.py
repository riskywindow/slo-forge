from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import asdict
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
    TrustedArtifactAnchor,
    TrustedClaimAnchor,
    TrustedEvidenceAnchor,
    ValidationContext,
    ValidationIssue,
    ValidationIssueCode,
    VerificationLevel,
    canonical_json,
    load_capsule,
    publish_capsule,
    seal_capsule,
    validate_capsule,
)
from sloforge.genesis.capsule.validator import (
    _performance_acceptance_failures,
    _validate_benchmark,
    _validate_quality_artifact,
    _validate_resource_artifact,
    _validate_runtime_bundle,
)
from sloforge.genesis.policy_dsl import compile_policy, parse_policy

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
    media_type: str = "application/json",
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
        media_type=media_type,
    )


def _runtime_bundle(
    *, candidate_hash: Digest, source_hash: Digest, tokenizer_hash: Digest
) -> tuple[bytes, dict[str, str]]:
    policy_source = (
        b"policy capsule_fixture\ninput queue_length int 0 8\noutput int 1 1\nlimit 8\nreturn 1\n"
    )
    policy_bytecode = json.dumps(
        asdict(compile_policy(parse_policy(policy_source.decode("utf-8")))),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    tested_config = {
        "genome_hash": candidate_hash.value,
        "package_hash": source_hash.value,
        "policy_bytecode_path": "policy.bytecode.json",
        "policy_bytecode_sha256": hashlib.sha256(policy_bytecode).hexdigest(),
        "reference_package_root": "/trusted/reference-package",
    }
    tested_config_payload = canonical_json(tested_config) + b"\n"
    packaged_config = dict(tested_config)
    packaged_config["reference_package_root"] = "reference_package"
    entries = {
        "runtime.py": b"raise SystemExit('sandbox launch required')\n",
        "correctness_harness.py": b"raise SystemExit('fixture only')\n",
        "deployment_manifest.json": b"{}\n",
        "runtime_config.json": canonical_json(packaged_config) + b"\n",
        "tested_runtime_config.json": tested_config_payload,
        "policy.slo": policy_source,
        "policy.bytecode.json": policy_bytecode,
        "reference_package/reference_package.json": b'{"tokenizer_module":"tokenizer.py"}\n',
        "reference_package/tokenizer.py": b"tokenizer",
    }
    assert hashlib.sha256(entries["reference_package/tokenizer.py"]).hexdigest() == (
        tokenizer_hash.value
    )
    manifest = {
        "candidate_genome_hash": candidate_hash.value,
        "direct_launch_supported": False,
        "entries": {
            name: hashlib.sha256(payload).hexdigest() for name, payload in sorted(entries.items())
        },
        "tested_runtime_config_sha256": hashlib.sha256(tested_config_payload).hexdigest(),
    }
    entries["bundle_manifest.json"] = canonical_json(manifest) + b"\n"
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in sorted(entries.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o444 << 16
            archive.writestr(info, payload)
    tested_hashes = {
        name: hashlib.sha256(entries[bundle_name]).hexdigest()
        for name, bundle_name in {
            "runtime.py": "runtime.py",
            "correctness_harness.py": "correctness_harness.py",
            "deployment_manifest.json": "deployment_manifest.json",
            "policy.bytecode.json": "policy.bytecode.json",
            "policy.slo": "policy.slo",
            "runtime_config.json": "tested_runtime_config.json",
        }.items()
    }
    return output.getvalue(), tested_hashes


def _complete_capsule(root: Path) -> tuple[GenesisCapsule, ValidationContext]:
    candidate_hash = _constant_digest("candidate")
    source_hash = _constant_digest("model")
    tokenizer_hash = _constant_digest("tokenizer")
    hardware_hash = _constant_digest("hardware")
    definition = _add_artifact(
        root,
        "definition",
        ArtifactRole.BENCHMARK_DEFINITION,
        json.dumps(
            {
                "execution_order": [
                    {"alternative": "baseline", "trial": 0},
                    {"alternative": "candidate", "trial": 1},
                    {"alternative": "candidate", "trial": 0},
                    {"alternative": "baseline", "trial": 1},
                ],
                "run_order_algorithm": "python-random-v1",
                "run_order_seed": 101,
                "warmup_iterations": 2,
                "bootstrap_rounds": 2000,
                "confidence": 0.95,
                "statistical_seed": 0,
            },
            sort_keys=True,
        ).encode(),
    )
    software = _add_artifact(root, "software", ArtifactRole.SOFTWARE_MANIFEST, b"{}")
    lock = _add_artifact(root, "dependency-lock", ArtifactRole.DEPENDENCY_LOCK, b"runtime==1.0.0\n")
    workload_hash = _constant_digest("workload")
    baseline_document = RawBenchmarkSamples(
        benchmark_definition_digest=definition.digest,
        workload_fingerprint=workload_hash,
        hardware_fingerprint=hardware_hash,
        software_manifest_digest=software.digest,
        samples=(
            RawBenchmarkSample(trial=0, seed=11, value=12.0, execution_ordinal=0),
            RawBenchmarkSample(trial=1, seed=12, value=14.0, execution_ordinal=3),
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
            RawBenchmarkSample(trial=0, seed=11, value=9.0, execution_ordinal=2),
            RawBenchmarkSample(trial=1, seed=12, value=11.0, execution_ordinal=1),
        ),
    )
    samples = _add_artifact(
        root,
        "samples",
        ArtifactRole.PERFORMANCE_SAMPLES,
        samples_document.model_dump_json().encode(),
        origin=ArtifactOrigin.PERFORMANCE_EVIDENCE,
    )
    runtime_payload, tested_runtime_hashes = _runtime_bundle(
        candidate_hash=candidate_hash,
        source_hash=source_hash,
        tokenizer_hash=tokenizer_hash,
    )
    runtime = _add_artifact(
        root,
        "runtime",
        ArtifactRole.GENERATED_RUNTIME,
        runtime_payload,
        origin=ArtifactOrigin.GENERATED_UNTRUSTED,
        media_type="application/zip",
    )
    deployment = _add_artifact(root, "deployment", ArtifactRole.DEPLOYMENT, b"deployment")
    rollback = _add_artifact(
        root,
        "rollback",
        ArtifactRole.ROLLBACK,
        b"rollback",
        origin=ArtifactOrigin.TRUSTED,
    )
    evidence_artifacts: dict[EvidenceClass, ArtifactRef] = {}
    role_by_class = {
        EvidenceClass.SEMANTIC: ArtifactRole.SEMANTIC_EVIDENCE,
        EvidenceClass.QUALITY: ArtifactRole.QUALITY_EVIDENCE,
        EvidenceClass.RESOURCE: ArtifactRole.RESOURCE_EVIDENCE,
        EvidenceClass.PERFORMANCE: ArtifactRole.PERFORMANCE_SAMPLES,
        EvidenceClass.OPERATIONAL: ArtifactRole.OPERATIONAL_EVIDENCE,
    }
    for evidence_class, role in role_by_class.items():
        payload = b"{}"
        if evidence_class is EvidenceClass.QUALITY:
            payload = json.dumps(
                {
                    "schema_version": "1.0.0",
                    "cases": [
                        {
                            "expected": [1],
                            "observed": [1],
                            "exact_match": True,
                        }
                    ],
                    "case_count": 1,
                    "observed": 1.0,
                    "threshold": 1.0,
                    "passed": True,
                    "runtime_artifact_hashes": tested_runtime_hashes,
                },
                sort_keys=True,
            ).encode()
        elif evidence_class is EvidenceClass.RESOURCE:
            single_runtime_bytes = 1024 + 792 + runtime.size_bytes
            capacity_bytes = 1024 * 1024
            payload = json.dumps(
                {
                    "schema_version": "1.1.0",
                    "maximum_prompt_tokens": 1,
                    "maximum_generated_tokens": 1,
                    "maximum_output_events_per_request": 2,
                    "persistent_state_bytes_per_request": 8,
                    "runtime_queue_depth": 1,
                    "bounded_request_bytes": 792,
                    "bounded_queue_bytes": 792,
                    "genome_declared_peak_host_bytes": 0,
                    "interpreter_and_model_reserve_bytes": 1024,
                    "runtime_bundle_bytes": runtime.size_bytes,
                    "single_runtime_peak_bytes": single_runtime_bytes,
                    "champion_challenger_coexistence_bytes": 2 * single_runtime_bytes,
                    "capacity_bytes": capacity_bytes,
                    "safety_margin_fraction": 0.2,
                    "usable_capacity_bytes": int(capacity_bytes * 0.8),
                    "reference_state_bytes_per_request": 8,
                    "genome_state_bytes_per_request": 8,
                    "genome_state_layouts": ["contiguous"],
                    "state_allocator_layout": "contiguous",
                    "state_allocator_page_bytes": 8,
                    "state_allocator_reserved_bytes_per_request": 8,
                    "state_allocator_total_bytes": 8,
                    "state_allocator_layout_matches_genome": True,
                    "state_allocator_bound_matches_genome": True,
                    "state_allocator_capacity_valid": True,
                    "maximum_processes": 2,
                    "passed": True,
                },
                sort_keys=True,
            ).encode()
        evidence_artifacts[evidence_class] = (
            samples
            if evidence_class is EvidenceClass.PERFORMANCE
            else _add_artifact(root, f"evidence-{evidence_class.value}", role, payload)
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
            artifact_ids=(
                (
                    definition.artifact_id,
                    samples.artifact_id,
                    software.artifact_id,
                    baseline.artifact_id,
                )
                if evidence_class is EvidenceClass.PERFORMANCE
                else (artifact.artifact_id,)
            ),
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
            source_model_hash=source_hash,
            tokenizer_hash=tokenizer_hash,
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
                noise_floor=0.05,
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
                    regression_probability=0.0,
                    practical_significance_threshold=0.1,
                ),
            ),
        ),
    )
    sealed = seal_capsule(capsule)
    assert sealed.capsule_digest is not None
    context = ValidationContext(
        expected_capsule_digest=sealed.capsule_digest,
        source_model_hash=source_hash,
        tokenizer_hash=tokenizer_hash,
        workload_contract_hash=workload_hash,
        hardware_contract_hash=hardware_hash,
        hardware_fingerprint=hardware_hash,
        hardware_architecture="cpu-test",
        device_count=1,
        dependency_lock_hash=lock.digest,
        dependencies=(CurrentDependency(name="runtime", version="1.0.0"),),
        trusted_evidence_anchors=tuple(
            TrustedEvidenceAnchor(
                evidence_id=record.evidence_id,
                evidence_record_digest=_digest(canonical_json(record)),
                issuer=record.issuer,
                issuer_version=record.issuer_version,
                artifacts=tuple(
                    TrustedArtifactAnchor(
                        artifact_id=artifact_id,
                        digest=next(
                            item.digest
                            for item in capsule.artifacts
                            if item.artifact_id == artifact_id
                        ),
                    )
                    for artifact_id in record.artifact_ids
                ),
            )
            for record in evidence
        ),
        trusted_claim_anchors=tuple(
            TrustedClaimAnchor(
                claim_id=claim.claim_id,
                claim_digest=_digest(canonical_json(claim)),
            )
            for claim in claims
        ),
        trusted_artifact_anchors=(
            TrustedArtifactAnchor(artifact_id=rollback.artifact_id, digest=rollback.digest),
        ),
        trusted_verifier_version="1.0.0",
        now=NOW,
    )
    return sealed, context


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


def test_quality_gate_rejects_nan_or_negative_threshold(tmp_path: Path) -> None:
    evidence = tmp_path / "quality.json"
    base = {
        "schema_version": "1.0.0",
        "cases": [{"expected": [1], "observed": [2], "exact_match": False}],
        "case_count": 1,
        "observed": 0.0,
        "passed": True,
    }
    for threshold in (-1.0, float("nan")):
        evidence.write_text(json.dumps({**base, "threshold": threshold}), encoding="utf-8")
        assert "finite probabilities" in (_validate_quality_artifact(evidence) or "")


def test_resource_gate_rejects_legacy_or_negative_bounds(tmp_path: Path) -> None:
    capsule, _context = _complete_capsule(tmp_path)
    resource = next(
        item for item in capsule.artifacts if item.role is ArtifactRole.RESOURCE_EVIDENCE
    )
    path = tmp_path / resource.path
    document = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**document, "schema_version": "1.0.0"}), encoding="utf-8")
    assert "unsupported schema" in (
        _validate_resource_artifact(path, runtime_bundle_bytes=None) or ""
    )
    path.write_text(json.dumps({**document, "runtime_queue_depth": -1}), encoding="utf-8")
    assert "out-of-domain" in (_validate_resource_artifact(path, runtime_bundle_bytes=None) or "")


def test_benchmark_gate_reconstructs_randomized_order_from_seed(tmp_path: Path) -> None:
    capsule, context = _complete_capsule(tmp_path)
    benchmark = capsule.benchmarks[0]
    artifacts = {item.artifact_id: item for item in capsule.artifacts}
    resolved = {item.artifact_id: tmp_path / item.path for item in capsule.artifacts}
    definition_path = resolved[benchmark.definition_artifact_id]
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    definition["execution_order"] = sorted(
        definition["execution_order"], key=lambda item: (item["alternative"], item["trial"])
    )
    definition_path.write_text(json.dumps(definition), encoding="utf-8")
    issues: list[ValidationIssue] = []

    _validate_benchmark(benchmark, artifacts, resolved, context, issues)

    assert any("deterministic run-order reconstruction" in issue.message for issue in issues)


def test_performance_claim_must_anchor_complete_benchmark_provenance(tmp_path: Path) -> None:
    capsule, context = _complete_capsule(tmp_path)
    performance = next(
        record for record in capsule.evidence if record.evidence_class is EvidenceClass.PERFORMANCE
    )
    candidate_samples = next(
        artifact
        for artifact in capsule.artifacts
        if artifact.artifact_id == capsule.benchmarks[0].raw_samples_artifact_id
    )
    incomplete = performance.model_copy(update={"artifact_ids": (candidate_samples.artifact_id,)})
    resealed = seal_capsule(
        capsule.model_copy(
            update={
                "capsule_digest": None,
                "evidence": tuple(
                    incomplete if record.evidence_id == performance.evidence_id else record
                    for record in capsule.evidence
                ),
            }
        )
    )
    assert resealed.capsule_digest is not None
    attacker_reanchored = TrustedEvidenceAnchor(
        evidence_id=incomplete.evidence_id,
        evidence_record_digest=_digest(canonical_json(incomplete)),
        issuer=incomplete.issuer,
        issuer_version=incomplete.issuer_version,
        artifacts=(
            TrustedArtifactAnchor(
                artifact_id=candidate_samples.artifact_id,
                digest=candidate_samples.digest,
            ),
        ),
    )
    report = validate_capsule(
        resealed,
        tmp_path,
        context.model_copy(
            update={
                "expected_capsule_digest": resealed.capsule_digest,
                "trusted_evidence_anchors": tuple(
                    attacker_reanchored if anchor.evidence_id == performance.evidence_id else anchor
                    for anchor in context.trusted_evidence_anchors
                ),
            }
        ),
    )

    codes = {issue.code for issue in report.issues}
    assert ValidationIssueCode.BENCHMARK_PROVENANCE_INVALID in codes
    assert ValidationIssueCode.EVIDENCE_UNTRUSTED not in codes
    assert not report.promotion_eligible


def test_promotion_rejects_runtime_outside_validated_bundle_format(tmp_path: Path) -> None:
    capsule, context = _complete_capsule(tmp_path)
    runtime = next(
        artifact
        for artifact in capsule.artifacts
        if artifact.role is ArtifactRole.GENERATED_RUNTIME
    )
    mislabeled = runtime.model_copy(update={"media_type": "application/octet-stream"})
    resealed = seal_capsule(
        capsule.model_copy(
            update={
                "capsule_digest": None,
                "artifacts": tuple(
                    mislabeled if artifact.artifact_id == runtime.artifact_id else artifact
                    for artifact in capsule.artifacts
                ),
            }
        )
    )
    assert resealed.capsule_digest is not None

    report = validate_capsule(
        resealed,
        tmp_path,
        context.model_copy(update={"expected_capsule_digest": resealed.capsule_digest}),
    )
    assert ValidationIssueCode.REQUIRED_ARTIFACT_MISSING in {issue.code for issue in report.issues}
    assert not report.promotion_eligible


def test_truthful_high_regression_probability_cannot_pass_performance_gate() -> None:
    summary = BenchmarkSummary(
        metric="latency",
        unit="milliseconds",
        objective="minimize",
        tail_quantile=0.95,
        median=50.0,
        tail_percentile=200.0,
        confidence_low=50.0,
        confidence_high=50.0,
        effect_size=0.5,
        regression_probability=2 / 7,
        practical_significance_threshold=0.1,
    )
    failures = _performance_acceptance_failures(
        summary,
        baseline_median=100.0,
        regression_probability=2 / 7,
        threshold=0.1,
    )
    assert "paired regression probability gate" in failures


def test_trusted_artifact_origin_requires_external_anchor(tmp_path: Path) -> None:
    capsule, context = _complete_capsule(tmp_path)
    unanchored = context.model_copy(update={"trusted_artifact_anchors": ()})
    report = validate_capsule(capsule, tmp_path, unanchored)
    assert ValidationIssueCode.EVIDENCE_UNTRUSTED in {item.code for item in report.issues}
    assert not report.local_evolution_eligible


def test_promotion_claim_scope_requires_external_anchor(tmp_path: Path) -> None:
    capsule, context = _complete_capsule(tmp_path)
    operational = next(
        claim for claim in capsule.claims if claim.category is ClaimCategory.OPERATIONAL
    )
    broadened = operational.model_copy(
        update={
            "scope": operational.scope.model_copy(
                update={
                    "hardware_fingerprints": (
                        *operational.scope.hardware_fingerprints,
                        _constant_digest("unverified-hardware"),
                    )
                }
            )
        }
    )
    resealed = seal_capsule(
        capsule.model_copy(
            update={
                "capsule_digest": None,
                "claims": tuple(
                    broadened if claim.claim_id == operational.claim_id else claim
                    for claim in capsule.claims
                ),
            }
        )
    )
    assert resealed.capsule_digest is not None
    report = validate_capsule(
        resealed,
        tmp_path,
        context.model_copy(update={"expected_capsule_digest": resealed.capsule_digest}),
    )

    assert ValidationIssueCode.EVIDENCE_UNTRUSTED in {issue.code for issue in report.issues}
    assert not report.promotion_eligible


def test_runtime_bundle_entry_count_is_bounded_before_extraction(tmp_path: Path) -> None:
    capsule, _context = _complete_capsule(tmp_path)
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index in range(4_097):
            archive.writestr(f"entry-{index:04d}", b"")
    path = tmp_path / "oversized-runtime.zip"
    path.write_bytes(payload.getvalue())
    artifact = ArtifactRef(
        artifact_id="oversized-runtime",
        role=ArtifactRole.GENERATED_RUNTIME,
        origin=ArtifactOrigin.GENERATED_UNTRUSTED,
        digest=_digest(payload.getvalue()),
        size_bytes=len(payload.getvalue()),
        path=path.name,
        media_type="application/zip",
    )
    issues: list[ValidationIssue] = []
    _validate_runtime_bundle(capsule, artifact, path, issues)
    assert len(issues) == 1
    assert issues[0].code is ValidationIssueCode.ARTIFACT_TAMPERED
    assert "entry count" in issues[0].message


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


def test_external_expected_digest_and_evidence_anchors_reject_forgery(tmp_path: Path) -> None:
    capsule, context = _complete_capsule(tmp_path)
    wrong_expected = context.model_copy(
        update={"expected_capsule_digest": _constant_digest("attacker-selected-capsule")}
    )
    assert ValidationIssueCode.MANIFEST_TAMPERED.value in _codes(capsule, tmp_path, wrong_expected)

    semantic = next(
        item for item in capsule.evidence if item.evidence_class is EvidenceClass.SEMANTIC
    )
    forged_record = semantic.model_copy(update={"issuer": EvidenceIssuer.TRUSTED_VALIDATOR})
    forged = seal_capsule(
        capsule.model_copy(
            update={
                "capsule_digest": None,
                "evidence": tuple(
                    forged_record if item.evidence_id == semantic.evidence_id else item
                    for item in capsule.evidence
                ),
            }
        )
    )
    assert forged.capsule_digest is not None
    attacker_expected_forgery = context.model_copy(
        update={"expected_capsule_digest": forged.capsule_digest}
    )
    assert ValidationIssueCode.EVIDENCE_UNTRUSTED.value in _codes(
        forged, tmp_path, attacker_expected_forgery
    )


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


def test_claim_level_cannot_be_borrowed_from_an_unrelated_evidence_class(
    tmp_path: Path,
) -> None:
    capsule, context = _complete_capsule(tmp_path)
    semantic = next(item for item in capsule.claims if item.category is ClaimCategory.SEMANTIC)
    elevated = semantic.model_copy(
        update={
            "level": VerificationLevel.HARDWARE_OPERATIONAL,
            "evidence_ids": (
                "evidence:semantic",
                "evidence:operational",
            ),
        }
    )
    resealed = seal_capsule(
        capsule.model_copy(
            update={
                "capsule_digest": None,
                "claims": tuple(
                    elevated if item.claim_id == semantic.claim_id else item
                    for item in capsule.claims
                ),
            }
        )
    )

    assert ValidationIssueCode.EVIDENCE_LEVEL_MISMATCH.value in _codes(resealed, tmp_path, context)


def test_evidence_cannot_mix_compatible_and_incompatible_artifact_roles(
    tmp_path: Path,
) -> None:
    capsule, context = _complete_capsule(tmp_path)
    semantic = next(
        item for item in capsule.evidence if item.evidence_class is EvidenceClass.SEMANTIC
    )
    altered = semantic.model_copy(
        update={
            "artifact_ids": (
                *semantic.artifact_ids,
                "generated-runtime",
            )
        }
    )
    resealed = seal_capsule(
        capsule.model_copy(
            update={
                "capsule_digest": None,
                "evidence": tuple(
                    altered if item.evidence_id == semantic.evidence_id else item
                    for item in capsule.evidence
                ),
            }
        )
    )

    assert ValidationIssueCode.EVIDENCE_INCOMPLETE.value in _codes(resealed, tmp_path, context)


@pytest.mark.parametrize(
    "summary_update",
    (
        {"median": 9.5},
        {"confidence_low": 9.1},
        {"regression_probability": 0.5},
    ),
)
def test_resealed_benchmark_with_altered_result_is_rejected(
    tmp_path: Path, summary_update: dict[str, float]
) -> None:
    capsule, context = _complete_capsule(tmp_path)
    benchmark = capsule.benchmarks[0]
    altered = benchmark.model_copy(
        update={"summary": benchmark.summary.model_copy(update=summary_update)}
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


def test_intermediate_symlink_inside_capsule_is_rejected(tmp_path: Path) -> None:
    capsule, context = _complete_capsule(tmp_path)
    artifacts = tmp_path / "artifacts"
    relocated = tmp_path / "relocated-artifacts"
    artifacts.rename(relocated)
    artifacts.symlink_to(relocated, target_is_directory=True)
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
