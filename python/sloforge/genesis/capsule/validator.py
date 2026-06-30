"""Independent, fail-closed Genesis capsule validator."""

from __future__ import annotations

import hashlib
import json
import math
import stat
import statistics
import zipfile
from pathlib import Path

from pydantic import ValidationError

from sloforge.genesis.policy_dsl import authenticate_bytecode_source, load_bytecode_document

from .canonical import calculate_capsule_digest, canonical_json
from .models import (
    ArtifactOrigin,
    ArtifactRef,
    ArtifactRole,
    BenchmarkEvidence,
    BenchmarkSummary,
    CapsuleValidationReport,
    ClaimCategory,
    CounterexampleCorpus,
    Digest,
    EvidenceClass,
    EvidenceIssuer,
    EvidenceResult,
    GenesisCapsule,
    RawBenchmarkSamples,
    ValidationContext,
    ValidationIssue,
    ValidationIssueCode,
    VerificationLevel,
    verification_level_rank,
)
from .statistics import bootstrap_median_interval, paired_regression_probability

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
_MAXIMUM_ARTIFACT_COUNT = 4_096
_MAXIMUM_ARTIFACT_BYTES = 256 * 1024 * 1024
_MAXIMUM_TOTAL_ARTIFACT_BYTES = 512 * 1024 * 1024
_MAXIMUM_RUNTIME_BUNDLE_ENTRIES = 4_096
_MAXIMUM_RUNTIME_BUNDLE_BYTES = 64 * 1024 * 1024
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
    EvidenceClass.BUILD: frozenset({ArtifactRole.COMPILED_BINARY, ArtifactRole.GENERATED_RUNTIME}),
    EvidenceClass.SEMANTIC: frozenset(
        {ArtifactRole.SEMANTIC_EVIDENCE, ArtifactRole.DIFFERENTIAL_TEST_RESULT}
    ),
    EvidenceClass.QUALITY: frozenset({ArtifactRole.QUALITY_EVIDENCE}),
    EvidenceClass.RESOURCE: frozenset({ArtifactRole.RESOURCE_EVIDENCE}),
    EvidenceClass.PERFORMANCE: frozenset({ArtifactRole.PERFORMANCE_SAMPLES}),
    EvidenceClass.OPERATIONAL: frozenset(
        {
            ArtifactRole.OPERATIONAL_EVIDENCE,
            ArtifactRole.MODEL_CHECK_RESULT,
            ArtifactRole.ROLLBACK,
            ArtifactRole.STATE_CONVERSION,
        }
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

    try:
        if root.is_symlink():
            return None
        candidate = root
        for component in artifact.path.split("/"):
            candidate = candidate / component
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


def _validate_runtime_bundle(
    capsule: GenesisCapsule,
    artifact: ArtifactRef,
    path: Path,
    issues: list[ValidationIssue],
) -> None:
    prefix = f"artifacts.{artifact.artifact_id}"
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if not 1 <= len(members) <= _MAXIMUM_RUNTIME_BUNDLE_ENTRIES:
                raise ValueError("bundle entry count exceeds the trusted validation bound")
            total_uncompressed = sum(member.file_size for member in members)
            if total_uncompressed > _MAXIMUM_RUNTIME_BUNDLE_BYTES:
                raise ValueError("bundle uncompressed size exceeds the trusted validation bound")
            names = [member.filename for member in members]
            if len(names) != len(set(names)) or any(
                name.startswith("/") or ".." in Path(name).parts for name in names
            ):
                raise ValueError("bundle contains duplicate or unsafe paths")
            if any(
                member.is_dir() or stat.S_ISLNK(member.external_attr >> 16) for member in members
            ):
                raise ValueError("bundle contains a directory or symlink entry")
            required = {
                "runtime.py",
                "runtime_config.json",
                "tested_runtime_config.json",
                "correctness_harness.py",
                "deployment_manifest.json",
                "policy.slo",
                "policy.bytecode.json",
                "bundle_manifest.json",
                "reference_package/reference_package.json",
            }
            if not required.issubset(names):
                raise ValueError(f"bundle is missing {sorted(required - set(names))}")
            manifest = json.loads(archive.read("bundle_manifest.json"))
            config = json.loads(archive.read("runtime_config.json"))
            tested_config = json.loads(archive.read("tested_runtime_config.json"))
            package_manifest = json.loads(archive.read("reference_package/reference_package.json"))
            declared = manifest["entries"]
            actual_entries = set(names) - {"bundle_manifest.json"}
            if set(declared) != actual_entries:
                raise ValueError("bundle manifest does not cover every dependency")
            for name, digest in declared.items():
                if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                    raise ValueError(f"bundle entry digest mismatch: {name}")
            if manifest["candidate_genome_hash"] != capsule.identity.candidate_genome_hash.value:
                raise ValueError("bundle candidate genome does not match capsule identity")
            if config["genome_hash"] != capsule.identity.candidate_genome_hash.value:
                raise ValueError("runtime config genome does not match capsule identity")
            if manifest.get("direct_launch_supported") is not False:
                raise ValueError("runtime bundle permits an untrusted direct launch")
            if hashlib.sha256(
                archive.read("tested_runtime_config.json")
            ).hexdigest() != manifest.get("tested_runtime_config_sha256"):
                raise ValueError("tested runtime configuration digest mismatch")
            expected_config = dict(tested_config)
            expected_config["reference_package_root"] = "reference_package"
            if config != expected_config:
                raise ValueError("packaged runtime configuration is not an audited root rewrite")
            if config["package_hash"] != capsule.identity.source_model_hash.value:
                raise ValueError("runtime source package does not match capsule identity")
            policy_path = str(config["policy_bytecode_path"])
            if policy_path not in actual_entries:
                raise ValueError("runtime policy dependency is absent")
            if (
                hashlib.sha256(archive.read(policy_path)).hexdigest()
                != config["policy_bytecode_sha256"]
            ):
                raise ValueError("runtime policy dependency digest mismatch")
            policy = load_bytecode_document(archive.read(policy_path))
            authenticate_bytecode_source(policy, archive.read("policy.slo"))
            tokenizer = f"reference_package/{package_manifest['tokenizer_module']}"
            if tokenizer not in actual_entries or hashlib.sha256(
                archive.read(tokenizer)
            ).hexdigest() != (capsule.identity.tokenizer_hash.value):
                raise ValueError("runtime tokenizer dependency does not match capsule identity")
    except (KeyError, OSError, TypeError, ValueError, zipfile.BadZipFile) as error:
        _append(
            issues,
            ValidationIssueCode.ARTIFACT_TAMPERED,
            prefix,
            f"generated runtime bundle is incomplete or inconsistent: {error}",
        )


def _validate_policy_artifacts(resolved: dict[str, Path], issues: list[ValidationIssue]) -> None:
    bytecode_path = resolved.get("generated-policy-bytecode")
    source_path = resolved.get("generated-policy")
    if bytecode_path is None or source_path is None:
        return
    try:
        program = load_bytecode_document(bytecode_path.read_bytes())
        authenticate_bytecode_source(program, source_path.read_bytes())
    except (OSError, ValueError) as error:
        _append(
            issues,
            ValidationIssueCode.ARTIFACT_TAMPERED,
            "artifacts.generated-policy-bytecode",
            f"generated policy is malformed or unauthenticated: {error}",
        )


def _validate_runtime_test_binding(
    capsule: GenesisCapsule,
    resolved: dict[str, Path],
    issues: list[ValidationIssue],
) -> None:
    """Bind packaged runtime bytes to the independently anchored differential run."""

    bundles = [
        artifact
        for artifact in capsule.artifacts
        if artifact.role is ArtifactRole.GENERATED_RUNTIME
        and artifact.media_type == "application/zip"
        and artifact.artifact_id in resolved
    ]
    quality_artifacts = [
        artifact
        for artifact in capsule.artifacts
        if artifact.role is ArtifactRole.QUALITY_EVIDENCE and artifact.artifact_id in resolved
    ]
    if len(bundles) != 1 or len(quality_artifacts) != 1:
        return
    try:
        quality = json.loads(resolved[quality_artifacts[0].artifact_id].read_text(encoding="utf-8"))
        tested_hashes = quality["runtime_artifact_hashes"]
        if not isinstance(tested_hashes, dict) or quality.get("passed") is not True:
            raise ValueError("quality evidence omits a passing tested-runtime manifest")
        entry_mapping = {
            "runtime.py": "runtime.py",
            "correctness_harness.py": "correctness_harness.py",
            "deployment_manifest.json": "deployment_manifest.json",
            "policy.bytecode.json": "policy.bytecode.json",
            "policy.slo": "policy.slo",
            "runtime_config.json": "tested_runtime_config.json",
        }
        with zipfile.ZipFile(resolved[bundles[0].artifact_id]) as archive:
            for tested_name, bundle_name in entry_mapping.items():
                expected = tested_hashes.get(tested_name)
                if not isinstance(expected, str):
                    raise ValueError(f"differential evidence omits {tested_name}")
                if hashlib.sha256(archive.read(bundle_name)).hexdigest() != expected:
                    raise ValueError(f"packaged {bundle_name} differs from tested {tested_name}")
    except (KeyError, OSError, TypeError, ValueError, zipfile.BadZipFile) as error:
        _append(
            issues,
            ValidationIssueCode.ARTIFACT_TAMPERED,
            "artifacts.generated-runtime",
            f"generated runtime is not the independently tested implementation: {error}",
        )


def _validate_quality_artifact(path: Path) -> str | None:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        cases = document["cases"]
        if not isinstance(cases, list) or not cases:
            return "quality evidence has no replayable cases"
        exact = 0
        for case in cases:
            if not isinstance(case, dict):
                return "quality case is not an object"
            reproduced = bool(case.get("expected") == case.get("observed"))
            if case.get("exact_match") is not reproduced:
                return "quality case exact-match flag is inconsistent"
            exact += reproduced
        observed = exact / len(cases)
        if document.get("case_count") != len(cases) or not math.isclose(
            float(document["observed"]), observed, rel_tol=0.0, abs_tol=0.0
        ):
            return "quality summary is not derived from its cases"
        if document.get("passed") is not True or observed < float(document["threshold"]):
            return "quality evidence does not satisfy its declared threshold"
    except (KeyError, OSError, TypeError, ValueError):
        return "quality evidence is not a valid replayable document"
    return None


def _validate_resource_artifact(path: Path, *, runtime_bundle_bytes: int | None) -> str | None:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        request_bytes = (
            (int(document["maximum_prompt_tokens"]) + int(document["maximum_generated_tokens"])) * 8
            + int(document["maximum_output_events_per_request"]) * 128
            + int(document["persistent_state_bytes_per_request"])
            + 512
        )
        queue_bytes = int(document["runtime_queue_depth"]) * request_bytes
        single = max(
            int(document["genome_declared_peak_host_bytes"]),
            int(document["interpreter_and_model_reserve_bytes"])
            + queue_bytes
            + int(document["runtime_bundle_bytes"]),
        )
        coexistence = single * 2
        usable = int(
            int(document["capacity_bytes"]) * (1.0 - float(document["safety_margin_fraction"]))
        )
        passed = usable >= coexistence
        if (
            int(document["bounded_request_bytes"]) != request_bytes
            or int(document["bounded_queue_bytes"]) != queue_bytes
            or int(document["single_runtime_peak_bytes"]) != single
            or int(document["champion_challenger_coexistence_bytes"]) != coexistence
            or int(document["usable_capacity_bytes"]) != usable
            or (
                runtime_bundle_bytes is not None
                and int(document["runtime_bundle_bytes"]) != runtime_bundle_bytes
            )
            or document.get("passed") is not passed
        ):
            return "resource summary is not derived from runtime and genome bounds"
    except (KeyError, OSError, TypeError, ValueError):
        return "resource evidence is not a valid replayable document"
    return None


def _performance_acceptance_failures(
    summary: BenchmarkSummary,
    *,
    baseline_median: float,
    regression_probability: float,
    threshold: float,
) -> tuple[str, ...]:
    """Apply conservative promotion gates to independently recomputed statistics."""

    failures: list[str] = []
    if regression_probability > 0.05:
        failures.append("paired regression probability gate")
    if summary.objective == "minimize":
        conservative_bound = baseline_median * (1.0 - threshold)
        if summary.confidence_high >= conservative_bound:
            failures.append("conservative improvement confidence gate")
    else:
        conservative_bound = baseline_median * (1.0 + threshold)
        if summary.confidence_low <= conservative_bound:
            failures.append("conservative improvement confidence gate")
    return tuple(failures)


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
    definition_path = resolved.get(benchmark.definition_artifact_id)
    raw_path = resolved.get(benchmark.raw_samples_artifact_id)
    baseline_path = resolved.get(benchmark.baseline_artifact_id)
    if definition_path is None or raw_path is None or baseline_path is None:
        return
    try:
        definition = json.loads(definition_path.read_text(encoding="utf-8"))
        samples = RawBenchmarkSamples.model_validate_json(raw_path.read_bytes(), strict=True)
        baseline_samples = RawBenchmarkSamples.model_validate_json(
            baseline_path.read_bytes(), strict=True
        )
    except (OSError, ValidationError, ValueError) as exc:
        _append(
            issues,
            ValidationIssueCode.BENCHMARK_PROVENANCE_INVALID,
            f"{prefix}.raw_samples",
            f"raw samples are not a valid evidence document: {exc}",
        )
        return
    definition_artifact = artifacts[benchmark.definition_artifact_id]
    software = artifacts[benchmark.software_manifest_artifact_id]
    mismatches: list[str] = []
    if len(samples.samples) != benchmark.sample_count:
        mismatches.append("sample count")
    if len(baseline_samples.samples) != benchmark.sample_count:
        mismatches.append("baseline sample count")
    if benchmark.repetitions != benchmark.sample_count:
        mismatches.append("repetition count")
    if samples.benchmark_definition_digest != definition_artifact.digest:
        mismatches.append("benchmark definition digest")
    if samples.software_manifest_digest != software.digest:
        mismatches.append("software manifest digest")
    if samples.workload_fingerprint != benchmark.workload_fingerprint:
        mismatches.append("workload fingerprint")
    if samples.hardware_fingerprint != benchmark.hardware_fingerprint:
        mismatches.append("hardware fingerprint")
    if baseline_samples.benchmark_definition_digest != definition_artifact.digest:
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
    candidate_by_trial = {sample.trial: sample for sample in samples.samples}
    baseline_by_trial = {sample.trial: sample for sample in baseline_samples.samples}
    candidate_trials = {trial: sample.seed for trial, sample in candidate_by_trial.items()}
    baseline_trials = {trial: sample.seed for trial, sample in baseline_by_trial.items()}
    if candidate_trials != baseline_trials:
        mismatches.append("paired trial or seed identity")
    expected_execution = {
        (alternative, trial)
        for trial in candidate_trials
        for alternative in ("baseline", "candidate")
    }
    execution_order = definition.get("execution_order") if isinstance(definition, dict) else None
    try:
        if not isinstance(execution_order, list):
            raise TypeError("execution order must be a list")
        observed_execution = [
            (str(item["alternative"]), int(item["trial"])) for item in execution_order
        ]
    except (KeyError, TypeError, ValueError):
        observed_execution = []
    if (
        len(observed_execution) != len(expected_execution)
        or set(observed_execution) != expected_execution
    ):
        mismatches.append("recorded randomized execution order")
    combined_samples = tuple(("baseline", sample) for sample in baseline_samples.samples) + tuple(
        ("candidate", sample) for sample in samples.samples
    )
    if any(sample.execution_ordinal is None for _alternative, sample in combined_samples):
        mismatches.append("sample execution ordinals")
    else:
        execution_ordinals = [
            sample.execution_ordinal
            for _alternative, sample in combined_samples
            if sample.execution_ordinal is not None
        ]
        if sorted(execution_ordinals) != list(range(len(combined_samples))):
            mismatches.append("sample execution ordinal coverage")
        else:
            sample_execution = [
                (alternative, sample.trial)
                for alternative, sample in sorted(
                    combined_samples,
                    key=lambda item: (
                        item[1].execution_ordinal if item[1].execution_ordinal is not None else -1
                    ),
                )
            ]
            if sample_execution != observed_execution:
                mismatches.append("sample execution order binding")
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
    try:
        bootstrap_rounds = int(definition["bootstrap_rounds"])
        confidence = float(definition["confidence"])
        statistical_seed = int(definition["statistical_seed"])
        expected_low, expected_high = bootstrap_median_interval(
            tuple(sample.value for sample in samples.samples),
            seed=statistical_seed,
            rounds=bootstrap_rounds,
            confidence=confidence,
        )
    except (KeyError, TypeError, ValueError):
        expected_low = expected_high = math.nan
    if not math.isclose(
        expected_low, benchmark.summary.confidence_low, rel_tol=1e-12, abs_tol=1e-12
    ) or not math.isclose(
        expected_high, benchmark.summary.confidence_high, rel_tol=1e-12, abs_tol=1e-12
    ):
        mismatches.append("reported confidence interval")
    expected_regression_probability = paired_regression_probability(
        tuple(baseline_by_trial[trial].value for trial in sorted(baseline_by_trial)),
        tuple(candidate_by_trial[trial].value for trial in sorted(candidate_by_trial)),
        objective=benchmark.summary.objective,
    )
    if not math.isclose(
        expected_regression_probability,
        benchmark.summary.regression_probability,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        mismatches.append("reported regression probability")
    threshold = max(
        benchmark.summary.practical_significance_threshold,
        benchmark.noise_floor,
    )
    if expected_effect <= threshold:
        mismatches.append("practical significance gate")
    mismatches.extend(
        _performance_acceptance_failures(
            benchmark.summary,
            baseline_median=baseline_median,
            regression_probability=expected_regression_probability,
            threshold=threshold,
        )
    )
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
    elif capsule.capsule_digest != context.expected_capsule_digest:
        _append(
            issues,
            ValidationIssueCode.MANIFEST_TAMPERED,
            "capsule_digest",
            "capsule digest does not match the externally supplied expected digest",
        )

    artifacts = {artifact.artifact_id: artifact for artifact in capsule.artifacts}
    if len(capsule.artifacts) > _MAXIMUM_ARTIFACT_COUNT:
        _append(
            issues,
            ValidationIssueCode.ARTIFACT_SIZE_MISMATCH,
            "artifacts",
            "artifact count exceeds the trusted validation bound",
        )
    trusted_artifact_anchors = {
        item.artifact_id: item.digest for item in context.trusted_artifact_anchors
    }
    resolved: dict[str, Path] = {}
    observed_artifact_bytes = 0
    for artifact in capsule.artifacts[:_MAXIMUM_ARTIFACT_COUNT]:
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
        observed_artifact_bytes += actual_size
        if actual_size != artifact.size_bytes:
            _append(
                issues,
                ValidationIssueCode.ARTIFACT_SIZE_MISMATCH,
                f"artifacts.{artifact.artifact_id}",
                f"declared {artifact.size_bytes} bytes but found {actual_size}",
            )
        if (
            actual_size > _MAXIMUM_ARTIFACT_BYTES
            or observed_artifact_bytes > _MAXIMUM_TOTAL_ARTIFACT_BYTES
        ):
            _append(
                issues,
                ValidationIssueCode.ARTIFACT_SIZE_MISMATCH,
                f"artifacts.{artifact.artifact_id}",
                "artifact content exceeds the trusted validation resource bound",
            )
            resolved.pop(artifact.artifact_id, None)
            continue
        if _sha256_file(path) != artifact.digest.value:
            _append(
                issues,
                ValidationIssueCode.ARTIFACT_TAMPERED,
                f"artifacts.{artifact.artifact_id}",
                "artifact content does not match its declared digest",
            )
        if (
            artifact.role is ArtifactRole.GENERATED_RUNTIME
            and artifact.media_type == "application/zip"
        ):
            _validate_runtime_bundle(capsule, artifact, path, issues)

        if artifact.origin is ArtifactOrigin.TRUSTED:
            anchored = trusted_artifact_anchors.get(artifact.artifact_id)
            if anchored != artifact.digest:
                _append(
                    issues,
                    ValidationIssueCode.EVIDENCE_UNTRUSTED,
                    f"artifacts.{artifact.artifact_id}.origin",
                    "trusted artifact origin is not bound by the external validation context",
                )

    rollback_artifacts = [
        artifact for artifact in capsule.artifacts if artifact.role is ArtifactRole.ROLLBACK
    ]
    for artifact in rollback_artifacts:
        if artifact.origin is not ArtifactOrigin.TRUSTED:
            _append(
                issues,
                ValidationIssueCode.EVIDENCE_UNTRUSTED,
                f"artifacts.{artifact.artifact_id}.origin",
                "rollback authority must be externally anchored trusted material",
            )

    _validate_policy_artifacts(resolved, issues)
    _validate_runtime_test_binding(capsule, resolved, issues)

    evidence = {record.evidence_id: record for record in capsule.evidence}
    trust_anchors = {item.evidence_id: item for item in context.trusted_evidence_anchors}
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
        for artifact_id in record.artifact_ids:
            path = resolved.get(artifact_id)
            if path is None:
                continue
            diagnostic = (
                _validate_quality_artifact(path)
                if record.evidence_class is EvidenceClass.QUALITY
                else _validate_resource_artifact(
                    path,
                    runtime_bundle_bytes=next(
                        (
                            artifact.size_bytes
                            for artifact in capsule.artifacts
                            if artifact.role is ArtifactRole.GENERATED_RUNTIME
                            and artifact.media_type == "application/zip"
                        ),
                        None,
                    ),
                )
                if record.evidence_class is EvidenceClass.RESOURCE
                else None
            )
            if diagnostic is not None:
                _append(
                    issues,
                    ValidationIssueCode.EVIDENCE_INCOMPLETE,
                    prefix,
                    diagnostic,
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
        incompatible_roles = referenced_roles.difference(
            _ARTIFACT_ROLE_BY_EVIDENCE[record.evidence_class]
        )
        if referenced_roles and incompatible_roles:
            _append(
                issues,
                ValidationIssueCode.EVIDENCE_INCOMPLETE,
                prefix,
                "evidence references an artifact incompatible with its evidence class",
            )

    for claim in capsule.claims:
        prefix = f"claims.{claim.claim_id}"
        records = [evidence[item] for item in claim.evidence_ids if item in evidence]
        matching_records = [
            item
            for item in records
            if item.evidence_class is _EVIDENCE_CLASS_BY_CLAIM[claim.category]
        ]
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
        if matching_records and max(
            verification_level_rank(item.level) for item in matching_records
        ) < verification_level_rank(claim.level):
            _append(
                issues,
                ValidationIssueCode.EVIDENCE_LEVEL_MISMATCH,
                prefix,
                "claim level exceeds its matching evidence-class level",
            )
        if records and not matching_records:
            _append(
                issues,
                ValidationIssueCode.EVIDENCE_INCOMPLETE,
                prefix,
                "claim category is not supported by matching evidence",
            )
        if claim.promotion_required:
            for record in matching_records:
                anchor = trust_anchors.get(record.evidence_id)
                anchored_artifacts = (
                    {item.artifact_id: item.digest for item in anchor.artifacts}
                    if anchor is not None
                    else {}
                )
                referenced_artifacts = {
                    artifact_id: artifacts[artifact_id].digest
                    for artifact_id in record.artifact_ids
                    if artifact_id in artifacts
                }
                record_digest = Digest(value=hashlib.sha256(canonical_json(record)).hexdigest())
                if (
                    anchor is None
                    or anchor.evidence_record_digest != record_digest
                    or anchor.issuer is not record.issuer
                    or anchor.issuer_version != record.issuer_version
                    or anchored_artifacts != referenced_artifacts
                ):
                    _append(
                        issues,
                        ValidationIssueCode.EVIDENCE_UNTRUSTED,
                        f"{prefix}.evidence.{record.evidence_id}",
                        "promotion evidence is not bound to an externally trusted record and artifact digest",
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
    promotion_claims = tuple(claim for claim in capsule.claims if claim.promotion_required)
    promotion_level = (
        min(promotion_claims, key=lambda claim: verification_level_rank(claim.level)).level
        if promotion_claims
        else None
    )
    local_evolution_eligible = not issues
    production_categories = {ClaimCategory.PERFORMANCE, ClaimCategory.OPERATIONAL}
    production_claims = tuple(
        claim for claim in promotion_claims if claim.category in production_categories
    )
    external_production_eligible = bool(
        local_evolution_eligible
        and {claim.category for claim in production_claims} == production_categories
        and all(
            claim.level is VerificationLevel.HARDWARE_OPERATIONAL for claim in production_claims
        )
    )
    return CapsuleValidationReport(
        capsule_digest=capsule.capsule_digest,
        candidate_genome_hash=capsule.identity.candidate_genome_hash,
        promotion_verification_level=promotion_level,
        integrity_valid=integrity_valid,
        contract_compatible=contract_compatible,
        evidence_complete=evidence_complete,
        promotion_eligible=external_production_eligible,
        checked_at=context.now,
        issues=tuple(issues),
        local_evolution_eligible=local_evolution_eligible,
        external_production_eligible=external_production_eligible,
    )
