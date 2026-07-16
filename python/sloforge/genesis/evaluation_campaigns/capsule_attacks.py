"""Artifact-backed, multi-seed H6 capsule promotion attack campaign.

The campaign drives the production capsule validator with a promotion-complete
conformance fixture.  The fixture's benchmark values are deterministic trust-
boundary test vectors, not measurements and not hardware performance evidence.
Every attack is persisted, validated, then independently rebuilt and replayed.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import shutil
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self, TypedDict

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from sloforge.genesis.capsule import (
    ArtifactOrigin,
    ArtifactRef,
    ArtifactRole,
    BenchmarkEvidence,
    BenchmarkSummary,
    CapsuleIdentity,
    CapsuleValidationReport,
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
    ValidationIssueCode,
    VerificationLevel,
    canonical_json,
    seal_capsule,
    validate_capsule,
)
from sloforge.genesis.capsule.statistics import bootstrap_median_interval
from sloforge.genesis.policy_dsl import compile_policy, parse_policy

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonNegativeInt = Annotated[int, Field(ge=0)]

_OBSERVED = datetime(2026, 7, 1, tzinfo=UTC)
_VALID_UNTIL = datetime(2027, 7, 1, tzinfo=UTC)
_CHECKED_AT = datetime(2026, 8, 2, tzinfo=UTC)
_STALE_CHECKED_AT = datetime(2030, 1, 1, tzinfo=UTC)


class CapsuleAttackCampaignValidationError(ValueError):
    """H6 artifacts are missing, altered, or fail independent replay."""


class _Model(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        allow_inf_nan=False,
    )


class _ExecutionEntry(TypedDict):
    alternative: str
    trial: int


class AttackKind(StrEnum):
    MODIFIED_RUNTIME_ARTIFACT = "modified_runtime_artifact"
    HARDWARE_FINGERPRINT_MISMATCH = "hardware_fingerprint_mismatch"
    DEPENDENCY_VERSION_MISMATCH = "dependency_version_mismatch"
    STALE_EVIDENCE = "stale_evidence"
    INCOMPLETE_EVIDENCE = "incomplete_evidence"
    ALTERED_BENCHMARK_SUMMARY = "altered_benchmark_summary"
    ALTERED_QUALITY_EVIDENCE = "altered_quality_evidence"
    MISSING_COUNTEREXAMPLE_CORPUS = "missing_counterexample_corpus"
    INVALID_MODELCHECK_SCOPE = "invalid_modelcheck_scope"
    INCOMPATIBLE_STATE_MIGRATION = "incompatible_state_migration"


_ATTACKS = tuple(AttackKind)
_EXPECTED_CODES: dict[AttackKind, tuple[ValidationIssueCode, ...]] = {
    AttackKind.MODIFIED_RUNTIME_ARTIFACT: (ValidationIssueCode.ARTIFACT_TAMPERED,),
    AttackKind.HARDWARE_FINGERPRINT_MISMATCH: (ValidationIssueCode.HARDWARE_MISMATCH,),
    AttackKind.DEPENDENCY_VERSION_MISMATCH: (ValidationIssueCode.DEPENDENCY_MISMATCH,),
    AttackKind.STALE_EVIDENCE: (ValidationIssueCode.EVIDENCE_STALE,),
    AttackKind.INCOMPLETE_EVIDENCE: (
        ValidationIssueCode.EVIDENCE_INCOMPLETE,
        ValidationIssueCode.REQUIRED_EVIDENCE_CLASS_MISSING,
    ),
    AttackKind.ALTERED_BENCHMARK_SUMMARY: (ValidationIssueCode.BENCHMARK_PROVENANCE_INVALID,),
    AttackKind.ALTERED_QUALITY_EVIDENCE: (ValidationIssueCode.ARTIFACT_TAMPERED,),
    AttackKind.MISSING_COUNTEREXAMPLE_CORPUS: (ValidationIssueCode.COUNTEREXAMPLE_CORPUS_MISSING,),
    AttackKind.INVALID_MODELCHECK_SCOPE: (ValidationIssueCode.CLAIM_SCOPE_MISMATCH,),
    AttackKind.INCOMPATIBLE_STATE_MIGRATION: (ValidationIssueCode.ARTIFACT_TAMPERED,),
}
_DESCRIPTIONS: dict[AttackKind, str] = {
    AttackKind.MODIFIED_RUNTIME_ARTIFACT: (
        "change generated-runtime bytes after the manifest and trust context are sealed"
    ),
    AttackKind.HARDWARE_FINGERPRINT_MISMATCH: (
        "validate the capsule on a CPU fingerprint outside its declared hardware scope"
    ),
    AttackKind.DEPENDENCY_VERSION_MISMATCH: (
        "validate with an installed runtime version different from the capsule lock"
    ),
    AttackKind.STALE_EVIDENCE: "advance trusted time beyond every evidence validity horizon",
    AttackKind.INCOMPLETE_EVIDENCE: (
        "remove quality evidence while retaining the promotion-required quality claim"
    ),
    AttackKind.ALTERED_BENCHMARK_SUMMARY: (
        "reseal a lower benchmark median that is not derivable from immutable raw samples"
    ),
    AttackKind.ALTERED_QUALITY_EVIDENCE: (
        "change the replayable quality cases without changing their anchored artifact digest"
    ),
    AttackKind.MISSING_COUNTEREXAMPLE_CORPUS: (
        "remove the required counterexample corpus from a resealed manifest"
    ),
    AttackKind.INVALID_MODELCHECK_SCOPE: (
        "move the operational model-check claim to a hardware scope not containing the validator"
    ),
    AttackKind.INCOMPATIBLE_STATE_MIGRATION: (
        "replace the anchored state-conversion artifact with an incompatible source genome"
    ),
}


class CampaignScope(_Model):
    validator: Literal["sloforge.genesis.capsule.validator.validate_capsule"] = (
        "sloforge.genesis.capsule.validator.validate_capsule"
    )
    fixture: Literal["promotion_complete_capsule_conformance_vector_v1"] = (
        "promotion_complete_capsule_conformance_vector_v1"
    )
    evidence_scope: Literal["deterministic_local_cpu_validator_conformance"] = (
        "deterministic_local_cpu_validator_conformance"
    )
    benchmark_values: Literal["synthetic_validator_test_vectors_not_measurements"] = (
        "synthetic_validator_test_vectors_not_measurements"
    )
    hardware_backed_runs: Literal[0] = 0
    gpu_hours: Literal[0] = 0
    performance_claims: Literal[False] = False
    deployment_performed: Literal[False] = False


class ChangedFile(_Model):
    relative_path: NonEmpty
    before_sha256: Sha256
    after_sha256: Sha256


class AttackMutationRecord(_Model):
    schema_version: Literal["sloforge.genesis.h6-mutation/v1"] = "sloforge.genesis.h6-mutation/v1"
    seed: NonNegativeInt
    attack: AttackKind
    description: NonEmpty
    changed_manifest_paths: tuple[NonEmpty, ...]
    changed_context_paths: tuple[NonEmpty, ...]
    changed_files: tuple[ChangedFile, ...]
    resealed_manifest: bool
    attacker_selected_expected_digest: bool


class AttackObservation(_Model):
    seed: NonNegativeInt
    attack: AttackKind
    description: NonEmpty
    expected_issue_codes: tuple[ValidationIssueCode, ...]
    observed_issue_codes: tuple[ValidationIssueCode, ...]
    baseline_capsule_digest: Sha256
    attacked_capsule_digest: Sha256
    mutation_path: NonEmpty
    mutation_sha256: Sha256
    attacked_capsule_path: NonEmpty
    attacked_capsule_sha256: Sha256
    attacked_context_path: NonEmpty
    attacked_context_sha256: Sha256
    validation_report_path: NonEmpty
    validation_report_sha256: Sha256
    promotion_eligible: Literal[False]
    rejected: Literal[True]

    @model_validator(mode="after")
    def expected_detection_is_present(self) -> Self:
        if not set(self.expected_issue_codes).issubset(self.observed_issue_codes):
            raise ValueError("attack did not produce all required validator issue codes")
        return self


class SeedResult(_Model):
    seed: NonNegativeInt
    baseline_capsule_path: NonEmpty
    baseline_capsule_sha256: Sha256
    baseline_context_path: NonEmpty
    baseline_context_sha256: Sha256
    baseline_validation_path: NonEmpty
    baseline_validation_sha256: Sha256
    baseline_capsule_digest: Sha256
    baseline_promotion_eligible: Literal[True]
    attacks: tuple[AttackObservation, ...]

    @model_validator(mode="after")
    def complete_attack_matrix(self) -> Self:
        if tuple(item.attack for item in self.attacks) != _ATTACKS:
            raise ValueError("seed result does not contain the complete ordered attack matrix")
        if any(item.seed != self.seed for item in self.attacks):
            raise ValueError("attack seed differs from its seed result")
        return self


class CapsuleAttackCampaignReport(_Model):
    schema_version: Literal["sloforge.genesis.h6-campaign/v1"] = "sloforge.genesis.h6-campaign/v1"
    hypothesis_id: Literal["H6"] = "H6"
    statement: Literal[
        "Proof-carrying promotion catches unsafe or incompatible generated artifacts."
    ] = "Proof-carrying promotion catches unsafe or incompatible generated artifacts."
    seeds: tuple[NonNegativeInt, ...]
    scope: CampaignScope
    raw_results_path: NonEmpty
    raw_results_sha256: Sha256
    results: tuple[SeedResult, ...]
    attack_count: NonNegativeInt
    rejected_attack_count: NonNegativeInt
    issue_code_coverage: tuple[ValidationIssueCode, ...]
    conclusion: Literal["supported_in_declared_fixture_scope", "not_supported"]
    limitations: tuple[NonEmpty, ...]

    @model_validator(mode="after")
    def complete_campaign(self) -> Self:
        if len(self.seeds) < 3 or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("H6 requires at least three unique seeds")
        if tuple(item.seed for item in self.results) != self.seeds:
            raise ValueError("H6 seed result order differs from declared seeds")
        expected_count = len(self.seeds) * len(_ATTACKS)
        if self.attack_count != expected_count:
            raise ValueError("H6 attack count is inconsistent")
        if self.rejected_attack_count > self.attack_count:
            raise ValueError("H6 rejected count exceeds attack count")
        expected_conclusion = (
            "supported_in_declared_fixture_scope"
            if self.rejected_attack_count == self.attack_count
            else "not_supported"
        )
        if self.conclusion != expected_conclusion:
            raise ValueError("H6 conclusion is inconsistent")
        return self


@dataclass(frozen=True)
class _Fixture:
    capsule: GenesisCapsule
    context: ValidationContext
    capsule_path: Path
    context_path: Path
    validation_path: Path


@dataclass(frozen=True)
class _AttackMaterialization:
    mutation: AttackMutationRecord
    capsule: GenesisCapsule
    context: ValidationContext
    report: CapsuleValidationReport
    mutation_path: Path
    capsule_path: Path
    context_path: Path
    validation_path: Path


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(payload: bytes) -> Digest:
    return Digest(value=_sha256_bytes(payload))


def _named_digest(label: str) -> Digest:
    return _digest(label.encode("utf-8"))


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite H6 artifact: {path}") from error


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
    _write_once(path, payload)
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
    candidate_hash: Digest, source_hash: Digest, tokenizer_hash: Digest
) -> tuple[bytes, dict[str, str]]:
    policy_source = (
        b"policy h6_fixture\ninput queue_length int 0 8\noutput int 1 1\nlimit 8\nreturn 1\n"
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
        "reference_package/tokenizer.py": b"h6-tokenizer",
    }
    if (
        hashlib.sha256(entries["reference_package/tokenizer.py"]).hexdigest()
        != tokenizer_hash.value
    ):
        raise CapsuleAttackCampaignValidationError("H6 tokenizer fixture identity is inconsistent")
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


def _execution_order(seed: int) -> tuple[_ExecutionEntry, ...]:
    entries: list[_ExecutionEntry] = [
        {"alternative": alternative, "trial": trial}
        for alternative in ("baseline", "candidate")
        for trial in range(2)
    ]
    random.Random(seed).shuffle(entries)
    return tuple(entries)


def _build_fixture(root: Path, seed: int) -> _Fixture:
    candidate_hash = _named_digest(f"h6-candidate:{seed}")
    hardware_hash = _named_digest("h6-local-cpu-fingerprint")
    workload_hash = _named_digest("h6-workload-contract")
    order = _execution_order(seed)
    ordinals = {(item["alternative"], item["trial"]): ordinal for ordinal, item in enumerate(order)}
    definition = _add_artifact(
        root,
        "benchmark-definition",
        ArtifactRole.BENCHMARK_DEFINITION,
        json.dumps(
            {
                "fixture_only": True,
                "values_are_measurements": False,
                "execution_order": order,
                "run_order_algorithm": "python-random-v1",
                "run_order_seed": seed,
                "warmup_iterations": 1,
                "bootstrap_rounds": 256,
                "confidence": 0.95,
                "statistical_seed": seed,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"),
    )
    software = _add_artifact(
        root,
        "software-manifest",
        ArtifactRole.SOFTWARE_MANIFEST,
        canonical_json({"fixture": "h6", "hardware_backed": False}),
    )
    lock = _add_artifact(
        root,
        "dependency-lock",
        ArtifactRole.DEPENDENCY_LOCK,
        b"runtime==1.0.0\n",
    )
    baseline_document = RawBenchmarkSamples(
        benchmark_definition_digest=definition.digest,
        workload_fingerprint=workload_hash,
        hardware_fingerprint=hardware_hash,
        software_manifest_digest=software.digest,
        samples=(
            RawBenchmarkSample(
                trial=0,
                seed=seed,
                value=12.0,
                execution_ordinal=ordinals[("baseline", 0)],
            ),
            RawBenchmarkSample(
                trial=1,
                seed=seed + 1,
                value=14.0,
                execution_ordinal=ordinals[("baseline", 1)],
            ),
        ),
    )
    baseline = _add_artifact(
        root,
        "baseline-samples",
        ArtifactRole.PERFORMANCE_SAMPLES,
        canonical_json(baseline_document),
        origin=ArtifactOrigin.PERFORMANCE_EVIDENCE,
    )
    candidate_document = RawBenchmarkSamples(
        benchmark_definition_digest=definition.digest,
        workload_fingerprint=workload_hash,
        hardware_fingerprint=hardware_hash,
        software_manifest_digest=software.digest,
        samples=(
            RawBenchmarkSample(
                trial=0,
                seed=seed,
                value=9.0,
                execution_ordinal=ordinals[("candidate", 0)],
            ),
            RawBenchmarkSample(
                trial=1,
                seed=seed + 1,
                value=11.0,
                execution_ordinal=ordinals[("candidate", 1)],
            ),
        ),
    )
    samples = _add_artifact(
        root,
        "candidate-samples",
        ArtifactRole.PERFORMANCE_SAMPLES,
        canonical_json(candidate_document),
        origin=ArtifactOrigin.PERFORMANCE_EVIDENCE,
    )
    runtime_payload, tested_runtime_hashes = _runtime_bundle(
        candidate_hash, _named_digest("h6-model"), _named_digest("h6-tokenizer")
    )
    runtime = _add_artifact(
        root,
        "generated-runtime",
        ArtifactRole.GENERATED_RUNTIME,
        runtime_payload,
        origin=ArtifactOrigin.GENERATED_UNTRUSTED,
        media_type="application/zip",
    )
    deployment = _add_artifact(
        root,
        "deployment",
        ArtifactRole.DEPLOYMENT,
        canonical_json({"mode": "isolated-fixture", "live": False}),
    )
    rollback = _add_artifact(
        root,
        "rollback",
        ArtifactRole.ROLLBACK,
        canonical_json({"action": "restore-fixture-champion"}),
        origin=ArtifactOrigin.TRUSTED,
    )
    state_conversion = _add_artifact(
        root,
        "state-conversion",
        ArtifactRole.STATE_CONVERSION,
        canonical_json(
            {
                "source_genome_hash": candidate_hash.value,
                "target_genome_hash": candidate_hash.value,
                "compatible": True,
            }
        ),
        origin=ArtifactOrigin.GENERATED_UNTRUSTED,
    )
    semantic = _add_artifact(
        root,
        "semantic-evidence",
        ArtifactRole.SEMANTIC_EVIDENCE,
        canonical_json({"candidate_genome_hash": candidate_hash.value, "passed": True}),
    )
    quality = _add_artifact(
        root,
        "quality-evidence",
        ArtifactRole.QUALITY_EVIDENCE,
        canonical_json(
            {
                "schema_version": "1.0.0",
                "cases": [{"expected": [1], "observed": [1], "exact_match": True}],
                "case_count": 1,
                "observed": 1.0,
                "threshold": 1.0,
                "passed": True,
                "runtime_artifact_hashes": tested_runtime_hashes,
            }
        ),
    )
    resource = _add_artifact(
        root,
        "resource-evidence",
        ArtifactRole.RESOURCE_EVIDENCE,
        canonical_json(
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
                "single_runtime_peak_bytes": 1024 + 792 + runtime.size_bytes,
                "champion_challenger_coexistence_bytes": 2 * (1024 + 792 + runtime.size_bytes),
                "capacity_bytes": 1024 * 1024,
                "safety_margin_fraction": 0.2,
                "usable_capacity_bytes": int(1024 * 1024 * 0.8),
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
            }
        ),
    )
    modelcheck = _add_artifact(
        root,
        "model-check-result",
        ArtifactRole.MODEL_CHECK_RESULT,
        canonical_json(
            {
                "model": "promotion-state-migration",
                "bounds": {"requests": 2, "workers": 1},
                "invariants": ["one state owner", "rollback restores champion"],
                "result": "pass",
                "candidate_genome_hash": candidate_hash.value,
            }
        ),
        origin=ArtifactOrigin.FORMAL_OR_BOUNDED_EVIDENCE,
    )
    corpus = _add_artifact(
        root,
        "counterexample-corpus",
        ArtifactRole.COUNTEREXAMPLE_CORPUS,
        canonical_json(
            CounterexampleCorpus(
                candidate_genome_hash=candidate_hash,
                counterexample_artifact_ids=(),
                searched_domains=("bounded capsule promotion fixture",),
            )
        ),
    )

    artifacts_by_class: dict[EvidenceClass, tuple[ArtifactRef, ...]] = {
        EvidenceClass.SEMANTIC: (semantic,),
        EvidenceClass.QUALITY: (quality,),
        EvidenceClass.RESOURCE: (resource,),
        EvidenceClass.PERFORMANCE: (definition, samples, software, baseline),
        EvidenceClass.OPERATIONAL: (modelcheck, state_conversion),
    }
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
            artifact_ids=tuple(item.artifact_id for item in evidence_artifacts),
            observed_at=_OBSERVED,
            valid_until=_VALID_UNTIL,
            deterministic_seed=seed,
            assumptions=("isolated validator conformance fixture",),
        )
        for evidence_class, evidence_artifacts in artifacts_by_class.items()
    )
    class_by_category = {
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
            statement=f"validator conformance fixture {category.value} claim",
            scope=ClaimScope(
                input_domain=("bounded validator conformance vector",),
                hardware_fingerprints=(hardware_hash,),
                assumptions=("not evidence about deployed or hardware performance",),
            ),
            level=(
                VerificationLevel.HARDWARE_OPERATIONAL
                if category in {ClaimCategory.PERFORMANCE, ClaimCategory.OPERATIONAL}
                else VerificationLevel.PROPERTY
            ),
            result=EvidenceResult.PASS,
            evidence_ids=(f"evidence:{evidence_class.value}",),
        )
        for category, evidence_class in class_by_category.items()
    )
    hardware = HardwareCompatibility(
        hardware_contract_hash=hardware_hash,
        allowed_fingerprints=(hardware_hash,),
        architectures=("cpu-local-fixture",),
        restrictions=("validator conformance only", "not deployable evidence"),
    )
    low, high = bootstrap_median_interval((9.0, 11.0), seed=seed, rounds=256, confidence=0.95)
    capsule = seal_capsule(
        GenesisCapsule(
            identity=CapsuleIdentity(
                candidate_genome_hash=candidate_hash,
                source_model_hash=_named_digest("h6-model"),
                tokenizer_hash=_named_digest("h6-tokenizer"),
                workload_contract_hash=workload_hash,
                hardware_contract_hash=hardware_hash,
                compiler_version="1.0.0",
                verifier_version="1.0.0",
                git_commit="h6-conformance-fixture",
                dependency_lock_hash=lock.digest,
                generated_at=_OBSERVED,
            ),
            artifacts=(
                definition,
                software,
                lock,
                baseline,
                samples,
                runtime,
                deployment,
                rollback,
                state_conversion,
                semantic,
                quality,
                resource,
                modelcheck,
                corpus,
            ),
            dependencies=(DependencyRequirement(name="runtime", version="1.0.0"),),
            hardware=hardware,
            evidence=evidence,
            claims=claims,
            benchmarks=(
                BenchmarkEvidence(
                    benchmark_id="validator-vector",
                    definition_artifact_id=definition.artifact_id,
                    raw_samples_artifact_id=samples.artifact_id,
                    software_manifest_artifact_id=software.artifact_id,
                    baseline_artifact_id=baseline.artifact_id,
                    workload_fingerprint=workload_hash,
                    hardware_fingerprint=hardware_hash,
                    sample_count=2,
                    warmup_iterations=1,
                    repetitions=2,
                    randomized_run_order=True,
                    noise_floor=0.05,
                    summary=BenchmarkSummary(
                        metric="validator-test-vector",
                        unit="dimensionless",
                        objective="minimize",
                        tail_quantile=0.9,
                        median=10.0,
                        tail_percentile=10.8,
                        confidence_low=low,
                        confidence_high=high,
                        effect_size=3.0 / 13.0,
                        regression_probability=0.0,
                        practical_significance_threshold=0.1,
                    ),
                ),
            ),
            known_unsupported_cases=("all use outside validator conformance testing",),
            unverified_assumptions=("synthetic benchmark vectors are not measurements",),
        )
    )
    if capsule.capsule_digest is None:
        raise RuntimeError("sealed H6 fixture capsule has no digest")
    context = ValidationContext(
        expected_capsule_digest=capsule.capsule_digest,
        source_model_hash=capsule.identity.source_model_hash,
        tokenizer_hash=capsule.identity.tokenizer_hash,
        workload_contract_hash=workload_hash,
        hardware_contract_hash=hardware_hash,
        hardware_fingerprint=hardware_hash,
        hardware_architecture="cpu-local-fixture",
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
        now=_CHECKED_AT,
        require_promotion_evidence=True,
    )
    capsule_path = root / "capsule.json"
    context_path = root / "validation-context.json"
    validation_path = root / "baseline-validation.json"
    _write_once(capsule_path, canonical_json(capsule) + b"\n")
    _write_once(context_path, canonical_json(context) + b"\n")
    report = validate_capsule(capsule, root, context)
    if not report.promotion_eligible or report.issues:
        codes = ",".join(item.code.value for item in report.issues)
        raise CapsuleAttackCampaignValidationError(
            f"H6 baseline conformance capsule did not validate: {codes}"
        )
    _write_once(validation_path, canonical_json(report) + b"\n")
    return _Fixture(
        capsule=capsule,
        context=context,
        capsule_path=capsule_path,
        context_path=context_path,
        validation_path=validation_path,
    )


def _changed_file(
    baseline_root: Path, attack_root: Path, relative_path: str, replacement: bytes
) -> ChangedFile:
    baseline = baseline_root / relative_path
    attacked = attack_root / relative_path
    before = _sha256_file(baseline)
    attacked.write_bytes(replacement)
    return ChangedFile(
        relative_path=relative_path,
        before_sha256=before,
        after_sha256=_sha256_file(attacked),
    )


def _reseal(capsule: GenesisCapsule) -> GenesisCapsule:
    resealed = seal_capsule(capsule.model_copy(update={"capsule_digest": None}))
    if resealed.capsule_digest is None:
        raise RuntimeError("resealed attack capsule has no digest")
    return resealed


def _materialize_attack(
    *,
    baseline_root: Path,
    fixture: _Fixture,
    attack_root: Path,
    seed: int,
    attack: AttackKind,
) -> _AttackMaterialization:
    attack_root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(baseline_root / "artifacts", attack_root / "artifacts")
    capsule = fixture.capsule
    context = fixture.context
    manifest_paths: tuple[str, ...] = ()
    context_paths: tuple[str, ...] = ()
    changed_files: tuple[ChangedFile, ...] = ()
    resealed = False
    attacker_digest = False

    if attack is AttackKind.MODIFIED_RUNTIME_ARTIFACT:
        artifact = next(
            item for item in capsule.artifacts if item.artifact_id == "generated-runtime"
        )
        original = (baseline_root / artifact.path).read_bytes()
        changed_files = (
            _changed_file(baseline_root, attack_root, artifact.path, original + b"tampered"),
        )
    elif attack is AttackKind.HARDWARE_FINGERPRINT_MISMATCH:
        context = context.model_copy(
            update={"hardware_fingerprint": _named_digest(f"different-hardware:{seed}")}
        )
        context_paths = ("hardware_fingerprint",)
    elif attack is AttackKind.DEPENDENCY_VERSION_MISMATCH:
        context = context.model_copy(
            update={"dependencies": (CurrentDependency(name="runtime", version="9.9.9"),)}
        )
        context_paths = ("dependencies.runtime.version",)
    elif attack is AttackKind.STALE_EVIDENCE:
        context = context.model_copy(update={"now": _STALE_CHECKED_AT})
        context_paths = ("now",)
    elif attack is AttackKind.INCOMPLETE_EVIDENCE:
        capsule = _reseal(
            capsule.model_copy(
                update={
                    "evidence": tuple(
                        item
                        for item in capsule.evidence
                        if item.evidence_class is not EvidenceClass.QUALITY
                    )
                }
            )
        )
        manifest_paths = ("evidence[evidence:quality]",)
        resealed = True
        attacker_digest = True
    elif attack is AttackKind.ALTERED_BENCHMARK_SUMMARY:
        benchmark = capsule.benchmarks[0]
        altered = benchmark.model_copy(
            update={"summary": benchmark.summary.model_copy(update={"median": 9.5})}
        )
        capsule = _reseal(capsule.model_copy(update={"benchmarks": (altered,)}))
        manifest_paths = ("benchmarks[validator-vector].summary.median",)
        resealed = True
        attacker_digest = True
    elif attack is AttackKind.ALTERED_QUALITY_EVIDENCE:
        artifact = next(
            item for item in capsule.artifacts if item.artifact_id == "quality-evidence"
        )
        changed_files = (
            _changed_file(
                baseline_root,
                attack_root,
                artifact.path,
                canonical_json(
                    {
                        "schema_version": "1.0.0",
                        "cases": [{"expected": [1], "observed": [2], "exact_match": False}],
                        "case_count": 1,
                        "observed": 0.0,
                        "threshold": 1.0,
                        "passed": False,
                    }
                ),
            ),
        )
    elif attack is AttackKind.MISSING_COUNTEREXAMPLE_CORPUS:
        capsule = _reseal(
            capsule.model_copy(
                update={
                    "artifacts": tuple(
                        item
                        for item in capsule.artifacts
                        if item.role is not ArtifactRole.COUNTEREXAMPLE_CORPUS
                    )
                }
            )
        )
        manifest_paths = ("artifacts[counterexample-corpus]",)
        resealed = True
        attacker_digest = True
    elif attack is AttackKind.INVALID_MODELCHECK_SCOPE:
        operational = next(
            item for item in capsule.claims if item.category is ClaimCategory.OPERATIONAL
        )
        altered_claim = operational.model_copy(
            update={
                "scope": operational.scope.model_copy(
                    update={
                        "hardware_fingerprints": (_named_digest(f"out-of-scope-modelcheck:{seed}"),)
                    }
                )
            }
        )
        capsule = _reseal(
            capsule.model_copy(
                update={
                    "claims": tuple(
                        altered_claim if item.claim_id == operational.claim_id else item
                        for item in capsule.claims
                    )
                }
            )
        )
        manifest_paths = ("claims[claim:operational].scope.hardware_fingerprints",)
        resealed = True
        attacker_digest = True
    elif attack is AttackKind.INCOMPATIBLE_STATE_MIGRATION:
        artifact = next(
            item for item in capsule.artifacts if item.artifact_id == "state-conversion"
        )
        changed_files = (
            _changed_file(
                baseline_root,
                attack_root,
                artifact.path,
                canonical_json(
                    {
                        "source_genome_hash": _named_digest(f"incompatible-source:{seed}").value,
                        "target_genome_hash": capsule.identity.candidate_genome_hash.value,
                        "compatible": False,
                    }
                ),
            ),
        )

    if attacker_digest:
        if capsule.capsule_digest is None:
            raise RuntimeError("attacked capsule has no digest")
        context = context.model_copy(update={"expected_capsule_digest": capsule.capsule_digest})
        context_paths = (*context_paths, "expected_capsule_digest")
    mutation = AttackMutationRecord(
        seed=seed,
        attack=attack,
        description=_DESCRIPTIONS[attack],
        changed_manifest_paths=manifest_paths,
        changed_context_paths=context_paths,
        changed_files=changed_files,
        resealed_manifest=resealed,
        attacker_selected_expected_digest=attacker_digest,
    )
    report = validate_capsule(capsule, attack_root, context)
    issue_codes = {item.code for item in report.issues}
    if report.promotion_eligible or not set(_EXPECTED_CODES[attack]).issubset(issue_codes):
        raise CapsuleAttackCampaignValidationError(
            f"H6 attack {attack.value} was not rejected by its required gate"
        )
    mutation_path = attack_root / "mutation.json"
    capsule_path = attack_root / "attacked-capsule.json"
    context_path = attack_root / "attacked-context.json"
    validation_path = attack_root / "validation-report.json"
    _write_once(mutation_path, canonical_json(mutation) + b"\n")
    _write_once(capsule_path, canonical_json(capsule) + b"\n")
    _write_once(context_path, canonical_json(context) + b"\n")
    _write_once(validation_path, canonical_json(report) + b"\n")
    return _AttackMaterialization(
        mutation=mutation,
        capsule=capsule,
        context=context,
        report=report,
        mutation_path=mutation_path,
        capsule_path=capsule_path,
        context_path=context_path,
        validation_path=validation_path,
    )


def _attack_observation(
    fixture: _Fixture,
    materialization: _AttackMaterialization,
    seed: int,
    attack: AttackKind,
) -> AttackObservation:
    if fixture.capsule.capsule_digest is None or materialization.capsule.capsule_digest is None:
        raise RuntimeError("H6 observation requires sealed capsules")
    return AttackObservation(
        seed=seed,
        attack=attack,
        description=_DESCRIPTIONS[attack],
        expected_issue_codes=_EXPECTED_CODES[attack],
        observed_issue_codes=tuple(
            sorted(
                {item.code for item in materialization.report.issues}, key=lambda item: item.value
            )
        ),
        baseline_capsule_digest=fixture.capsule.capsule_digest.value,
        attacked_capsule_digest=materialization.capsule.capsule_digest.value,
        mutation_path=str(materialization.mutation_path.resolve()),
        mutation_sha256=_sha256_file(materialization.mutation_path),
        attacked_capsule_path=str(materialization.capsule_path.resolve()),
        attacked_capsule_sha256=_sha256_file(materialization.capsule_path),
        attacked_context_path=str(materialization.context_path.resolve()),
        attacked_context_sha256=_sha256_file(materialization.context_path),
        validation_report_path=str(materialization.validation_path.resolve()),
        validation_report_sha256=_sha256_file(materialization.validation_path),
        promotion_eligible=False,
        rejected=True,
    )


def _seed_result(output_root: Path, seed: int) -> SeedResult:
    seed_root = output_root / "seeds" / str(seed)
    baseline_root = seed_root / "baseline"
    fixture = _build_fixture(baseline_root, seed)
    attacks: list[AttackObservation] = []
    for attack in _ATTACKS:
        materialization = _materialize_attack(
            baseline_root=baseline_root,
            fixture=fixture,
            attack_root=seed_root / "attacks" / attack.value,
            seed=seed,
            attack=attack,
        )
        attacks.append(_attack_observation(fixture, materialization, seed, attack))
    if fixture.capsule.capsule_digest is None:
        raise RuntimeError("H6 fixture has no capsule digest")
    return SeedResult(
        seed=seed,
        baseline_capsule_path=str(fixture.capsule_path.resolve()),
        baseline_capsule_sha256=_sha256_file(fixture.capsule_path),
        baseline_context_path=str(fixture.context_path.resolve()),
        baseline_context_sha256=_sha256_file(fixture.context_path),
        baseline_validation_path=str(fixture.validation_path.resolve()),
        baseline_validation_sha256=_sha256_file(fixture.validation_path),
        baseline_capsule_digest=fixture.capsule.capsule_digest.value,
        baseline_promotion_eligible=True,
        attacks=tuple(attacks),
    )


def run_capsule_attack_campaign(
    output_directory: Path,
    *,
    seeds: tuple[int, ...] = (17, 29, 73129),
) -> CapsuleAttackCampaignReport:
    """Run and persist all H6 promotion attacks for at least three seeds."""

    if len(seeds) < 3 or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise ValueError("H6 requires at least three unique non-negative seeds")
    output_directory.mkdir(parents=True, exist_ok=True)
    if any(output_directory.iterdir()):
        raise FileExistsError("H6 output directory must be empty")
    results = tuple(_seed_result(output_directory, seed) for seed in seeds)
    raw_path = output_directory / "raw" / "seed-results.jsonl"
    _write_once(raw_path, b"".join(canonical_json(item) + b"\n" for item in results))
    observations = tuple(attack for result in results for attack in result.attacks)
    issue_coverage = tuple(
        sorted(
            {code for attack in observations for code in attack.observed_issue_codes},
            key=lambda item: item.value,
        )
    )
    rejected_count = sum(attack.rejected for attack in observations)
    report = CapsuleAttackCampaignReport(
        seeds=seeds,
        scope=CampaignScope(),
        raw_results_path=str(raw_path.resolve()),
        raw_results_sha256=_sha256_file(raw_path),
        results=results,
        attack_count=len(observations),
        rejected_attack_count=rejected_count,
        issue_code_coverage=issue_coverage,
        conclusion=(
            "supported_in_declared_fixture_scope"
            if rejected_count == len(observations)
            else "not_supported"
        ),
        limitations=(
            "the promotion-complete capsule is a validator conformance fixture, not a deployable generated runtime",
            "benchmark values are deterministic statistical test vectors and are not measurements",
            "the model-check scope attack exercises scoped-claim compatibility, not model-checker soundness",
            "state migration is checked through anchored state-conversion integrity; semantic conversion replay is outside this validator",
            "all execution is local CPU validation; no GPU, live deployment, or hardware performance is claimed",
        ),
    )
    report_path = output_directory / "report.json"
    _write_once(report_path, canonical_json(report) + b"\n")
    validate_capsule_attack_campaign(report_path)
    return report


def _validate_file(path_value: str, expected_sha256: str) -> Path:
    path = Path(path_value)
    if not path.is_file() or _sha256_file(path) != expected_sha256:
        raise CapsuleAttackCampaignValidationError(f"H6 artifact digest mismatch: {path}")
    return path


def validate_capsule_attack_campaign(
    report: CapsuleAttackCampaignReport | Path,
) -> CapsuleAttackCampaignReport:
    """Reopen, revalidate, and independently rebuild every H6 attack."""

    try:
        value = (
            CapsuleAttackCampaignReport.model_validate_json(report.read_bytes(), strict=True)
            if isinstance(report, Path)
            else report
        )
    except (OSError, ValueError) as error:
        raise CapsuleAttackCampaignValidationError("H6 campaign report is invalid") from error
    raw_path = _validate_file(value.raw_results_path, value.raw_results_sha256)
    try:
        reopened = tuple(
            SeedResult.model_validate_json(line, strict=True)
            for line in raw_path.read_text(encoding="utf-8").splitlines()
            if line
        )
    except (OSError, ValueError) as error:
        raise CapsuleAttackCampaignValidationError("raw H6 results are invalid") from error
    if reopened != value.results:
        raise CapsuleAttackCampaignValidationError("H6 report differs from raw seed results")

    observed_issue_codes: set[ValidationIssueCode] = set()
    with tempfile.TemporaryDirectory(prefix="sloforge-h6-replay-") as temporary:
        replay_root = Path(temporary)
        for result in reopened:
            capsule_path = _validate_file(
                result.baseline_capsule_path, result.baseline_capsule_sha256
            )
            context_path = _validate_file(
                result.baseline_context_path, result.baseline_context_sha256
            )
            validation_path = _validate_file(
                result.baseline_validation_path, result.baseline_validation_sha256
            )
            try:
                stored_capsule = GenesisCapsule.model_validate_json(
                    capsule_path.read_bytes(), strict=True
                )
                stored_context = ValidationContext.model_validate_json(
                    context_path.read_bytes(), strict=True
                )
                stored_baseline_report = CapsuleValidationReport.model_validate_json(
                    validation_path.read_bytes(), strict=True
                )
            except ValueError as error:
                raise CapsuleAttackCampaignValidationError(
                    "H6 baseline fixture artifact is invalid"
                ) from error
            baseline_root = capsule_path.parent
            recomputed_baseline = validate_capsule(stored_capsule, baseline_root, stored_context)
            if (
                recomputed_baseline != stored_baseline_report
                or not recomputed_baseline.promotion_eligible
                or recomputed_baseline.issues
                or stored_capsule.capsule_digest is None
                or stored_capsule.capsule_digest.value != result.baseline_capsule_digest
            ):
                raise CapsuleAttackCampaignValidationError(
                    "H6 baseline no longer independently validates"
                )
            rebuilt_fixture = _build_fixture(
                replay_root / f"seed-{result.seed}" / "baseline", result.seed
            )
            if (
                rebuilt_fixture.capsule != stored_capsule
                or rebuilt_fixture.context != stored_context
            ):
                raise CapsuleAttackCampaignValidationError(
                    "H6 baseline fixture does not deterministically rebuild"
                )
            for observation in result.attacks:
                mutation_path = _validate_file(
                    observation.mutation_path, observation.mutation_sha256
                )
                attacked_capsule_path = _validate_file(
                    observation.attacked_capsule_path,
                    observation.attacked_capsule_sha256,
                )
                attacked_context_path = _validate_file(
                    observation.attacked_context_path,
                    observation.attacked_context_sha256,
                )
                attack_validation_path = _validate_file(
                    observation.validation_report_path,
                    observation.validation_report_sha256,
                )
                try:
                    stored_mutation = AttackMutationRecord.model_validate_json(
                        mutation_path.read_bytes(), strict=True
                    )
                    attacked_capsule = GenesisCapsule.model_validate_json(
                        attacked_capsule_path.read_bytes(), strict=True
                    )
                    attacked_context = ValidationContext.model_validate_json(
                        attacked_context_path.read_bytes(), strict=True
                    )
                    stored_attack_report = CapsuleValidationReport.model_validate_json(
                        attack_validation_path.read_bytes(), strict=True
                    )
                except ValueError as error:
                    raise CapsuleAttackCampaignValidationError(
                        "H6 attack artifact is invalid"
                    ) from error
                for changed in stored_mutation.changed_files:
                    stored_changed_path = attacked_capsule_path.parent / changed.relative_path
                    baseline_changed_path = baseline_root / changed.relative_path
                    if (
                        not stored_changed_path.is_file()
                        or _sha256_file(stored_changed_path) != changed.after_sha256
                        or not baseline_changed_path.is_file()
                        or _sha256_file(baseline_changed_path) != changed.before_sha256
                    ):
                        raise CapsuleAttackCampaignValidationError(
                            "H6 changed-file provenance does not match stored artifacts"
                        )
                recomputed_attack_report = validate_capsule(
                    attacked_capsule, attacked_capsule_path.parent, attacked_context
                )
                if recomputed_attack_report != stored_attack_report:
                    raise CapsuleAttackCampaignValidationError(
                        "H6 stored validator report does not recompute"
                    )
                replayed = _materialize_attack(
                    baseline_root=rebuilt_fixture.capsule_path.parent,
                    fixture=rebuilt_fixture,
                    attack_root=(
                        replay_root
                        / f"seed-{result.seed}"
                        / "replayed-attacks"
                        / observation.attack.value
                    ),
                    seed=result.seed,
                    attack=observation.attack,
                )
                if (
                    replayed.mutation != stored_mutation
                    or replayed.capsule != attacked_capsule
                    or replayed.context != attacked_context
                    or replayed.report != stored_attack_report
                ):
                    raise CapsuleAttackCampaignValidationError(
                        f"H6 attack {observation.attack.value} does not independently replay"
                    )
                recomputed_observation = _attack_observation(
                    rebuilt_fixture, replayed, result.seed, observation.attack
                )
                comparison_fields = (
                    "seed",
                    "attack",
                    "description",
                    "expected_issue_codes",
                    "observed_issue_codes",
                    "baseline_capsule_digest",
                    "attacked_capsule_digest",
                    "promotion_eligible",
                    "rejected",
                )
                if any(
                    getattr(recomputed_observation, field) != getattr(observation, field)
                    for field in comparison_fields
                ):
                    raise CapsuleAttackCampaignValidationError(
                        "H6 attack observation does not recompute"
                    )
                observed_issue_codes.update(observation.observed_issue_codes)
    if value.attack_count != len(reopened) * len(_ATTACKS):
        raise CapsuleAttackCampaignValidationError("H6 aggregate attack count changed")
    if value.rejected_attack_count != value.attack_count:
        raise CapsuleAttackCampaignValidationError("H6 contains an unrejected attack")
    if value.issue_code_coverage != tuple(
        sorted(observed_issue_codes, key=lambda item: item.value)
    ):
        raise CapsuleAttackCampaignValidationError("H6 issue-code coverage does not recompute")
    return value


__all__ = [
    "AttackKind",
    "AttackMutationRecord",
    "AttackObservation",
    "CampaignScope",
    "CapsuleAttackCampaignReport",
    "CapsuleAttackCampaignValidationError",
    "ChangedFile",
    "SeedResult",
    "run_capsule_attack_campaign",
    "validate_capsule_attack_campaign",
]
