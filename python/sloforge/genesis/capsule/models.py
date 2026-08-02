"""Strict, capsule-local types for proof-carrying Genesis artifacts.

These types deliberately do not import the mutable synthesis IR.  A capsule is
the stable trust-boundary projection of a candidate, not an alias for the
compiler's in-memory candidate representation.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Final, Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

CAPSULE_SCHEMA_VERSION: Final = "1.0.0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]


class CapsuleModel(BaseModel):
    """Immutable, strict base for every value crossing the trust boundary."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        allow_inf_nan=False,
    )


class Digest(CapsuleModel):
    algorithm: Literal["sha256"] = "sha256"
    value: str

    @field_validator("value")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("sha256 digest must be 64 lowercase hexadecimal characters")
        return value


class VerificationLevel(StrEnum):
    BUILD = "level_0_build"
    DIFFERENTIAL = "level_1_differential"
    PROPERTY = "level_2_property"
    BOUNDED_EXHAUSTIVE = "level_3_bounded_exhaustive"
    SOLVER_BACKED = "level_4_solver_backed"
    HARDWARE_OPERATIONAL = "level_5_hardware_operational"


class ClaimCategory(StrEnum):
    BUILD = "build"
    SEMANTIC = "semantic"
    QUALITY = "quality"
    RESOURCE = "resource"
    PERFORMANCE = "performance"
    OPERATIONAL = "operational"


class EvidenceClass(StrEnum):
    BUILD = "build"
    SEMANTIC = "semantic"
    QUALITY = "quality"
    RESOURCE = "resource"
    PERFORMANCE = "performance"
    OPERATIONAL = "operational"
    MODEL_CHECK = "model_check"
    PROPERTY_TEST = "property_test"
    FUZZ = "fuzz"
    DIFFERENTIAL = "differential"


class EvidenceResult(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


class EvidenceIssuer(StrEnum):
    TRUSTED_VALIDATOR = "trusted_validator"
    OPERATOR_VERIFIER = "operator_verifier"
    QUALITY_HARNESS = "quality_harness"
    RESOURCE_ANALYZER = "resource_analyzer"
    BENCHMARK_HARNESS = "benchmark_harness"
    MODEL_CHECKER = "model_checker"
    PROPERTY_HARNESS = "property_harness"
    FUZZ_HARNESS = "fuzz_harness"
    SANDBOX = "sandbox"


class ArtifactRole(StrEnum):
    SOURCE_CODE = "source_code"
    COMPILED_BINARY = "compiled_binary"
    GENERATED_RUNTIME = "generated_runtime"
    GENERATED_POLICY = "generated_policy"
    GENERATED_KERNEL = "generated_kernel"
    DEPLOYMENT = "deployment"
    ROLLBACK = "rollback"
    STATE_CONVERSION = "state_conversion"
    SEMANTIC_EVIDENCE = "semantic_evidence"
    QUALITY_EVIDENCE = "quality_evidence"
    RESOURCE_EVIDENCE = "resource_evidence"
    PERFORMANCE_SAMPLES = "performance_samples"
    BENCHMARK_DEFINITION = "benchmark_definition"
    SOFTWARE_MANIFEST = "software_manifest"
    DEPENDENCY_LOCK = "dependency_lock"
    OPERATIONAL_EVIDENCE = "operational_evidence"
    COUNTEREXAMPLE_CORPUS = "counterexample_corpus"
    MODEL_CHECK_RESULT = "model_check_result"
    PROPERTY_TEST_RESULT = "property_test_result"
    DIFFERENTIAL_TEST_RESULT = "differential_test_result"
    FUZZ_RESULT = "fuzz_result"


class ArtifactOrigin(StrEnum):
    TRUSTED = "trusted"
    GENERATED_UNTRUSTED = "generated_untrusted"
    EXTERNAL_RUNTIME = "external_runtime"
    VERIFIED_EVIDENCE = "verified_evidence"
    PERFORMANCE_EVIDENCE = "performance_evidence"
    FORMAL_OR_BOUNDED_EVIDENCE = "formal_or_bounded_evidence"


class ArtifactRef(CapsuleModel):
    artifact_id: NonEmptyString
    role: ArtifactRole
    origin: ArtifactOrigin
    digest: Digest
    size_bytes: NonNegativeInt
    path: NonEmptyString
    media_type: NonEmptyString
    executable: bool = False

    @field_validator("artifact_id")
    @classmethod
    def validate_artifact_id(cls, value: str) -> str:
        if _SAFE_IDENTIFIER.fullmatch(value) is None:
            raise ValueError("artifact_id contains unsupported characters")
        return value

    @field_validator("path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        normalized = value.replace("\\", "/")
        parts = normalized.split("/")
        if (
            value != normalized
            or normalized.startswith("/")
            or any(part in {"", ".", ".."} for part in parts)
        ):
            raise ValueError("artifact path must be a normalized, safe relative path")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", normalized) is None:
            raise ValueError("artifact path contains unsupported characters")
        return value


class ClaimScope(CapsuleModel):
    input_domain: tuple[NonEmptyString, ...]
    shape_domain: tuple[NonEmptyString, ...] = ()
    dtype_domain: tuple[NonEmptyString, ...] = ()
    hardware_fingerprints: tuple[Digest, ...] = ()
    dependency_requirements: tuple[NonEmptyString, ...] = ()
    assumptions: tuple[NonEmptyString, ...] = ()
    exclusions: tuple[NonEmptyString, ...] = ()

    @model_validator(mode="after")
    def require_input_domain(self) -> Self:
        if not self.input_domain:
            raise ValueError("claim scope must declare a non-empty input domain")
        return self


class EvidenceRecord(CapsuleModel):
    evidence_id: NonEmptyString
    evidence_class: EvidenceClass
    level: VerificationLevel
    result: EvidenceResult
    issuer: EvidenceIssuer
    issuer_version: NonEmptyString
    artifact_ids: tuple[NonEmptyString, ...]
    observed_at: AwareDatetime
    valid_until: AwareDatetime | None
    deterministic_seed: int | None = None
    assumptions: tuple[NonEmptyString, ...] = ()

    @field_validator("evidence_id")
    @classmethod
    def validate_evidence_id(cls, value: str) -> str:
        if _SAFE_IDENTIFIER.fullmatch(value) is None:
            raise ValueError("evidence_id contains unsupported characters")
        return value

    @model_validator(mode="after")
    def validate_record(self) -> Self:
        if not self.artifact_ids:
            raise ValueError("evidence must reference at least one immutable artifact")
        if len(self.artifact_ids) != len(set(self.artifact_ids)):
            raise ValueError("evidence artifact references must be unique")
        if self.valid_until is not None and self.valid_until <= self.observed_at:
            raise ValueError("valid_until must be after observed_at")
        return self


class ScopedClaim(CapsuleModel):
    claim_id: NonEmptyString
    category: ClaimCategory
    statement: NonEmptyString
    scope: ClaimScope
    level: VerificationLevel
    result: EvidenceResult
    evidence_ids: tuple[NonEmptyString, ...]
    promotion_required: bool = True

    @field_validator("claim_id")
    @classmethod
    def validate_claim_id(cls, value: str) -> str:
        if _SAFE_IDENTIFIER.fullmatch(value) is None:
            raise ValueError("claim_id contains unsupported characters")
        return value

    @model_validator(mode="after")
    def require_evidence(self) -> Self:
        if not self.evidence_ids:
            raise ValueError("every scoped claim must reference evidence")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("claim evidence references must be unique")
        if (
            self.level is VerificationLevel.HARDWARE_OPERATIONAL
            and not self.scope.hardware_fingerprints
        ):
            raise ValueError("hardware/operational claims require an exact hardware scope")
        return self


class DependencyRequirement(CapsuleModel):
    name: NonEmptyString
    version: NonEmptyString
    package_digest: Digest | None = None


class HardwareCompatibility(CapsuleModel):
    hardware_contract_hash: Digest
    allowed_fingerprints: tuple[Digest, ...]
    architectures: tuple[NonEmptyString, ...]
    minimum_device_count: PositiveInt = 1
    restrictions: tuple[NonEmptyString, ...] = ()

    @model_validator(mode="after")
    def validate_hardware(self) -> Self:
        if not self.allowed_fingerprints:
            raise ValueError("at least one exact hardware fingerprint is required")
        if not self.architectures:
            raise ValueError("at least one hardware architecture is required")
        return self


class BenchmarkSummary(CapsuleModel):
    metric: NonEmptyString
    unit: NonEmptyString
    objective: Literal["minimize", "maximize"]
    tail_quantile: Annotated[float, Field(gt=0.5, lt=1.0)]
    median: NonNegativeFloat
    tail_percentile: NonNegativeFloat
    confidence_low: NonNegativeFloat
    confidence_high: NonNegativeFloat
    effect_size: float
    regression_probability: Annotated[float, Field(ge=0.0, le=1.0)]
    practical_significance_threshold: NonNegativeFloat

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if self.confidence_low > self.confidence_high:
            raise ValueError("confidence interval lower bound exceeds upper bound")
        if not self.confidence_low <= self.median <= self.confidence_high:
            raise ValueError("median must lie within the declared confidence interval")
        return self


class BenchmarkEvidence(CapsuleModel):
    benchmark_id: NonEmptyString
    definition_artifact_id: NonEmptyString
    raw_samples_artifact_id: NonEmptyString
    software_manifest_artifact_id: NonEmptyString
    baseline_artifact_id: NonEmptyString
    workload_fingerprint: Digest
    hardware_fingerprint: Digest
    sample_count: Annotated[int, Field(ge=2)]
    warmup_iterations: PositiveInt
    repetitions: Annotated[int, Field(ge=2)]
    randomized_run_order: bool
    noise_floor: NonNegativeFloat
    summary: BenchmarkSummary

    @model_validator(mode="after")
    def validate_artifact_roles_are_separate(self) -> Self:
        artifact_ids = (
            self.definition_artifact_id,
            self.raw_samples_artifact_id,
            self.software_manifest_artifact_id,
            self.baseline_artifact_id,
        )
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("benchmark provenance artifacts must be distinct")
        return self


class CapsuleIdentity(CapsuleModel):
    capsule_schema_version: Literal["1.0.0"] = CAPSULE_SCHEMA_VERSION
    candidate_genome_hash: Digest
    source_model_hash: Digest
    tokenizer_hash: Digest
    workload_contract_hash: Digest
    hardware_contract_hash: Digest
    compiler_version: NonEmptyString
    verifier_version: NonEmptyString
    git_commit: NonEmptyString
    dependency_lock_hash: Digest
    generated_at: AwareDatetime
    parent_capsule: Digest | None = None


class GenesisCapsule(CapsuleModel):
    """Hash-addressed manifest presented to the independent promotion gate."""

    identity: CapsuleIdentity
    capsule_digest: Digest | None = None
    artifacts: tuple[ArtifactRef, ...]
    dependencies: tuple[DependencyRequirement, ...]
    hardware: HardwareCompatibility
    evidence: tuple[EvidenceRecord, ...]
    claims: tuple[ScopedClaim, ...]
    benchmarks: tuple[BenchmarkEvidence, ...]
    known_unsupported_cases: tuple[NonEmptyString, ...] = ()
    unverified_assumptions: tuple[NonEmptyString, ...] = ()

    @model_validator(mode="after")
    def validate_unique_identifiers(self) -> Self:
        collections = (
            ("artifact", [item.artifact_id for item in self.artifacts]),
            ("artifact path", [item.path for item in self.artifacts]),
            ("evidence", [item.evidence_id for item in self.evidence]),
            ("claim", [item.claim_id for item in self.claims]),
            ("benchmark", [item.benchmark_id for item in self.benchmarks]),
            ("dependency", [item.name for item in self.dependencies]),
        )
        for label, identifiers in collections:
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(f"duplicate {label} identifier")
        if self.identity.hardware_contract_hash != self.hardware.hardware_contract_hash:
            raise ValueError("identity and compatibility hardware contract hashes differ")
        return self


class RawBenchmarkSample(CapsuleModel):
    trial: NonNegativeInt
    seed: int
    value: NonNegativeFloat
    execution_ordinal: NonNegativeInt | None = None


class RawBenchmarkSamples(CapsuleModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    benchmark_definition_digest: Digest
    workload_fingerprint: Digest
    hardware_fingerprint: Digest
    software_manifest_digest: Digest
    samples: tuple[RawBenchmarkSample, ...]

    @model_validator(mode="after")
    def validate_samples(self) -> Self:
        if len(self.samples) < 2:
            raise ValueError("raw benchmark evidence requires at least two samples")
        trials = [sample.trial for sample in self.samples]
        if len(trials) != len(set(trials)):
            raise ValueError("raw benchmark trial identifiers must be unique")
        return self


class CounterexampleCorpus(CapsuleModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    candidate_genome_hash: Digest
    counterexample_artifact_ids: tuple[NonEmptyString, ...]
    searched_domains: tuple[NonEmptyString, ...]

    @model_validator(mode="after")
    def declare_search_scope(self) -> Self:
        if not self.searched_domains:
            raise ValueError("counterexample corpus must declare the searched domains")
        return self


class CurrentDependency(CapsuleModel):
    name: NonEmptyString
    version: NonEmptyString
    package_digest: Digest | None = None


class TrustedArtifactAnchor(CapsuleModel):
    artifact_id: NonEmptyString
    digest: Digest


class TrustedEvidenceAnchor(CapsuleModel):
    evidence_id: NonEmptyString
    evidence_record_digest: Digest
    issuer: EvidenceIssuer
    issuer_version: NonEmptyString
    artifacts: tuple[TrustedArtifactAnchor, ...]

    @model_validator(mode="after")
    def validate_anchor(self) -> Self:
        artifact_ids = [item.artifact_id for item in self.artifacts]
        if not artifact_ids:
            raise ValueError("trusted evidence must anchor at least one artifact")
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("trusted artifact anchors must be unique")
        return self


class TrustedClaimAnchor(CapsuleModel):
    """External authority binding for a complete promotion claim and its scope."""

    claim_id: NonEmptyString
    claim_digest: Digest


class ValidationContext(CapsuleModel):
    expected_capsule_digest: Digest
    source_model_hash: Digest
    tokenizer_hash: Digest
    workload_contract_hash: Digest
    hardware_contract_hash: Digest
    hardware_fingerprint: Digest
    hardware_architecture: NonEmptyString
    device_count: PositiveInt
    dependency_lock_hash: Digest
    dependencies: tuple[CurrentDependency, ...]
    trusted_evidence_anchors: tuple[TrustedEvidenceAnchor, ...]
    trusted_claim_anchors: tuple[TrustedClaimAnchor, ...] = ()
    trusted_artifact_anchors: tuple[TrustedArtifactAnchor, ...] = ()
    trusted_verifier_version: NonEmptyString
    now: AwareDatetime
    require_promotion_evidence: bool = True

    @model_validator(mode="after")
    def validate_trust_anchors(self) -> Self:
        evidence_ids = [item.evidence_id for item in self.trusted_evidence_anchors]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("trusted evidence anchors must have unique evidence identifiers")
        claim_ids = [item.claim_id for item in self.trusted_claim_anchors]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("trusted claim anchors must have unique claim identifiers")
        artifact_ids = [item.artifact_id for item in self.trusted_artifact_anchors]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("trusted artifact anchors must have unique artifact identifiers")
        return self


class ValidationIssueCode(StrEnum):
    UNSEALED = "unsealed"
    MANIFEST_TAMPERED = "manifest_tampered"
    ARTIFACT_MISSING = "artifact_missing"
    ARTIFACT_TAMPERED = "artifact_tampered"
    ARTIFACT_SIZE_MISMATCH = "artifact_size_mismatch"
    UNSAFE_ARTIFACT_PATH = "unsafe_artifact_path"
    EVIDENCE_INCOMPLETE = "evidence_incomplete"
    EVIDENCE_STALE = "evidence_stale"
    EVIDENCE_FAILED = "evidence_failed"
    EVIDENCE_LEVEL_MISMATCH = "evidence_level_mismatch"
    EVIDENCE_UNTRUSTED = "evidence_untrusted"
    CLAIM_SCOPE_MISMATCH = "claim_scope_mismatch"
    CONTRACT_MISMATCH = "contract_mismatch"
    HARDWARE_MISMATCH = "hardware_mismatch"
    DEPENDENCY_MISSING = "dependency_missing"
    DEPENDENCY_MISMATCH = "dependency_mismatch"
    VERIFIER_MISMATCH = "verifier_mismatch"
    BENCHMARK_PROVENANCE_INVALID = "benchmark_provenance_invalid"
    COUNTEREXAMPLE_CORPUS_MISSING = "counterexample_corpus_missing"
    REQUIRED_ARTIFACT_MISSING = "required_artifact_missing"
    REQUIRED_EVIDENCE_CLASS_MISSING = "required_evidence_class_missing"


class ValidationIssue(CapsuleModel):
    code: ValidationIssueCode
    path: NonEmptyString
    message: NonEmptyString


class CapsuleValidationReport(CapsuleModel):
    capsule_digest: Digest | None
    candidate_genome_hash: Digest | None
    promotion_verification_level: VerificationLevel | None
    integrity_valid: bool
    contract_compatible: bool
    evidence_complete: bool
    promotion_eligible: bool
    checked_at: AwareDatetime
    issues: tuple[ValidationIssue, ...]
    local_evolution_eligible: bool = False
    external_production_eligible: bool = False


def verification_level_rank(level: VerificationLevel) -> int:
    """Return the ordered evidence rank without relying on enum spelling."""

    return {
        VerificationLevel.BUILD: 0,
        VerificationLevel.DIFFERENTIAL: 1,
        VerificationLevel.PROPERTY: 2,
        VerificationLevel.BOUNDED_EXHAUSTIVE: 3,
        VerificationLevel.SOLVER_BACKED: 4,
        VerificationLevel.HARDWARE_OPERATIONAL: 5,
    }[level]


def ensure_aware(value: datetime) -> datetime:
    """Narrow helper used by validation code and tests."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return value
