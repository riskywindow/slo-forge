"""Build a promotion-complete local Genesis capsule from persisted evidence only."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from sloforge.genesis.frontend import load_reference_package
from sloforge.genesis.ir import (
    CandidateSuccessState,
    Transformation,
    canonical_hash,
    load_candidate,
    load_inference_genome,
    load_transformation,
)
from sloforge.genesis.policy_dsl import authenticate_bytecode_source, load_bytecode_document
from sloforge.genesis.sandbox import SandboxLimits, SandboxRequest, execute_sandboxed
from sloforge.genesis.search import CandidateDesign
from sloforge.genesis.synthesis import (
    CancellationPolicyVerifier,
    CegisRunResult,
    ConstraintDocument,
    bounded_candidate_modelcheck_document,
    bounded_candidate_policy_property_document,
)
from sloforge.genesis.synthesis.lowering import lower_candidate

from .canonical import canonical_json, seal_capsule
from .io import publish_capsule
from .models import (
    ArtifactOrigin,
    ArtifactRef,
    ArtifactRole,
    CapsuleIdentity,
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
    ScopedClaim,
    TrustedArtifactAnchor,
    TrustedEvidenceAnchor,
    ValidationContext,
    VerificationLevel,
)
from .validator import validate_capsule


class CapsuleBuildResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    capsule_path: str
    capsule_digest: str
    context_path: str
    candidate_id: str
    artifact_count: int
    evidence_count: int
    claim_count: int
    promotion_eligible: bool
    local_evolution_eligible: bool
    external_production_eligible: bool
    performance_scope: str
    hardware_backed: bool


def _digest(payload: bytes) -> Digest:
    return Digest(value=hashlib.sha256(payload).hexdigest())


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        path.chmod(0o444)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite capsule artifact: {path}") from exc


def _artifact(
    root: Path,
    artifact_id: str,
    role: ArtifactRole,
    payload: bytes,
    *,
    origin: ArtifactOrigin,
    suffix: str = ".json",
    media_type: str = "application/json",
) -> ArtifactRef:
    relative = f"artifacts/{artifact_id}{suffix}"
    _write_once(root / relative, payload)
    return ArtifactRef(
        artifact_id=artifact_id,
        role=role,
        origin=origin,
        digest=_digest(payload),
        size_bytes=len(payload),
        path=relative,
        media_type=media_type,
    )


def _copy_artifact(
    root: Path,
    artifact_id: str,
    role: ArtifactRole,
    source: Path,
    *,
    origin: ArtifactOrigin,
    suffix: str,
    media_type: str,
) -> ArtifactRef:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"capsule source must be a regular non-symlink file: {source}")
    payload = source.read_bytes()
    result = _artifact(
        root,
        artifact_id,
        role,
        payload,
        origin=origin,
        suffix=suffix,
        media_type=media_type,
    )
    return result


def _runtime_bundle_bytes(
    run_directory: Path,
    candidate_directory: Path,
    package_root: Path,
    *,
    candidate_id: str,
    candidate_genome_hash: str,
) -> bytes:
    runtime_root = candidate_directory / "generated_runtime"
    policy = (candidate_directory / "policy.bytecode.json").read_bytes()
    policy_source = (candidate_directory / "policy.slo").read_bytes()
    admitted_policy = load_bytecode_document(policy)
    authenticate_bytecode_source(admitted_policy, policy_source)
    tested_config_payload = (runtime_root / "runtime_config.json").read_bytes()
    config = _read_object(runtime_root / "runtime_config.json")
    config.update(
        {
            "reference_package_root": "reference_package",
            "genome_hash": candidate_genome_hash,
            "policy_bytecode_path": "policy.bytecode.json",
            "policy_bytecode_sha256": hashlib.sha256(policy).hexdigest(),
        }
    )
    entries: dict[str, bytes] = {
        "runtime.py": (runtime_root / "runtime.py").read_bytes(),
        "correctness_harness.py": (runtime_root / "correctness_harness.py").read_bytes(),
        "deployment_manifest.json": (runtime_root / "deployment_manifest.json").read_bytes(),
        "runtime_config.json": canonical_json(config) + b"\n",
        "tested_runtime_config.json": tested_config_payload,
        "policy.slo": policy_source,
        "policy.bytecode.json": policy,
    }
    for path in sorted(package_root.rglob("*")):
        if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts:
            entries[f"reference_package/{path.relative_to(package_root).as_posix()}"] = (
                path.read_bytes()
            )
    manifest = {
        "schema_version": "1.0.0",
        "candidate_id": candidate_id,
        "candidate_genome_hash": candidate_genome_hash,
        "entries": {
            name: hashlib.sha256(payload).hexdigest() for name, payload in sorted(entries.items())
        },
        "trusted_launcher": "sloforge.genesis.sandbox.execute_sandboxed",
        "sandbox_argv": ["python", "runtime.py", "--seed", "<required>"],
        "direct_launch_supported": False,
        "tested_runtime_config_sha256": hashlib.sha256(tested_config_payload).hexdigest(),
    }
    entries["bundle_manifest.json"] = canonical_json(manifest) + b"\n"
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in sorted(entries.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o444 << 16
            archive.writestr(info, payload)
    return output.getvalue()


def _resource_evidence(
    runtime_config: dict[str, Any],
    genome: dict[str, Any],
    package_manifest: dict[str, Any],
    *,
    capacity_bytes: int,
    bundle_size_bytes: int,
) -> dict[str, object]:
    limits = runtime_config["limits"]
    queue_depth = int(limits["maximum_queue_depth"])
    prompt_tokens = int(limits["maximum_prompt_tokens"])
    generated_tokens = int(limits["maximum_generated_tokens"])
    output_events = int(limits["maximum_output_events_per_request"])
    dtype_bytes = {"int8": 1, "int32": 4, "int64": 8, "float32": 4, "float64": 8}
    state_bytes = 0
    for field in package_manifest["state_contract"]["fields"]:
        elements = 1
        for dimension in field["shape"]:
            elements *= int(dimension["maximum"])
        state_bytes += elements * dtype_bytes[str(field["dtype"])]
    declared_host_bytes = 0
    declared_queue_entries = 0

    def visit(value: object) -> None:
        nonlocal declared_host_bytes, declared_queue_entries
        if isinstance(value, dict):
            resource = value.get("resource_requirements")
            if isinstance(resource, dict):
                declared_host_bytes += int(resource.get("peak_host_bytes", 0))
                declared_queue_entries += int(resource.get("queue_entries", 0))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(genome)
    request_bytes = (prompt_tokens + generated_tokens) * 8 + output_events * 128 + state_bytes + 512
    queue_bytes = queue_depth * request_bytes
    interpreter_and_model_reserve = 128 * 1024**2
    single_runtime_peak = max(
        declared_host_bytes,
        interpreter_and_model_reserve + queue_bytes + bundle_size_bytes,
    )
    coexistence = single_runtime_peak * 2
    usable = int(capacity_bytes * 0.8)
    return {
        "schema_version": "1.0.0",
        "method": "runtime-config-genome-state-contract-upper-bound",
        "capacity_bytes": capacity_bytes,
        "usable_capacity_bytes": usable,
        "safety_margin_fraction": 0.20,
        "runtime_queue_depth": queue_depth,
        "genome_declared_queue_entries": declared_queue_entries,
        "maximum_prompt_tokens": prompt_tokens,
        "maximum_generated_tokens": generated_tokens,
        "maximum_output_events_per_request": output_events,
        "persistent_state_bytes_per_request": state_bytes,
        "bounded_request_bytes": request_bytes,
        "bounded_queue_bytes": queue_bytes,
        "runtime_bundle_bytes": bundle_size_bytes,
        "genome_declared_peak_host_bytes": declared_host_bytes,
        "interpreter_and_model_reserve_bytes": interpreter_and_model_reserve,
        "single_runtime_peak_bytes": single_runtime_peak,
        "champion_challenger_coexistence_bytes": coexistence,
        "maximum_processes": 2,
        "passed": usable >= coexistence,
        "unresolved_risk": "Python allocator behavior is covered only by the declared 20 percent margin",
    }


def _independent_runtime_differential(
    runtime_directory: Path,
    package_root: Path,
    final_corpus: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    """Replay the tested runtime and corpus through a fresh bounded sandbox."""

    repository_python = Path(__file__).resolve().parents[3]
    with tempfile.TemporaryDirectory(prefix="sloforge-capsule-replay-") as temporary:
        sandbox_output = Path(temporary) / "artifacts"
        result = execute_sandboxed(
            SandboxRequest(
                argv=(
                    sys.executable,
                    "correctness_harness.py",
                    "--samples",
                    str(final_corpus.resolve(strict=True)),
                    "--seed",
                    str(seed),
                    "--timeout-seconds",
                    "3",
                ),
                working_directory=runtime_directory.resolve(strict=True),
                read_only_paths=(
                    runtime_directory.resolve(strict=True),
                    package_root.resolve(strict=True),
                    repository_python,
                    Path(sys.prefix),
                    Path(sys.base_prefix),
                ),
                artifact_output_directory=sandbox_output,
                seed=seed,
                limits=SandboxLimits(
                    wall_time_seconds=15.0,
                    cpu_time_seconds=10,
                    memory_bytes=2 * 1024 * 1024 * 1024,
                    process_count=1,
                    output_bytes=64 * 1024,
                    artifact_bytes=1024 * 1024,
                    artifact_entries=16,
                    open_files=64,
                ),
            )
        )
        if not result.succeeded:
            raise ValueError(
                f"independent differential replay failed in the sandbox: {result.termination.value}"
            )
        try:
            document = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError) as error:
            raise ValueError("independent differential replay returned invalid JSON") from error
        if not isinstance(document, dict):
            raise ValueError("independent differential replay must return an object")
        cases = document.get("cases")
        if (
            document.get("passed") is not True
            or not isinstance(cases, list)
            or not cases
            or any(
                not isinstance(case, dict) or case.get("exact_match") is not True for case in cases
            )
        ):
            raise ValueError("independent differential replay did not pass exact matching")
        return document


def _git_commit(repository: Path) -> str:
    marker = repository / ".sloforge-source-commit"
    if marker.is_file() and not marker.is_symlink():
        value = marker.read_text(encoding="utf-8").strip()
        if 7 <= len(value) <= 64 and all(character in "0123456789abcdef" for character in value):
            return value
        raise ValueError(".sloforge-source-commit is not a lowercase Git object identifier")
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        timeout=5.0,
    ).stdout.strip()


def _validate_transformation_chain(
    transformation_paths: list[Path],
    *,
    transformation_ids: tuple[str, ...],
    baseline_genome_hash: str,
    candidate_genome_hash: str,
    trusted_transformations: tuple[Transformation, ...],
) -> list[tuple[Path, Transformation]]:
    if len(transformation_paths) != len(transformation_ids):
        raise ValueError("candidate transformation artifact set is incomplete")
    loaded_transformations = [(path, load_transformation(path)) for path in transformation_paths]
    transformations_by_id = {
        transformation.transformation_id: (path, transformation)
        for path, transformation in loaded_transformations
    }
    if len(transformations_by_id) != len(loaded_transformations) or set(
        transformations_by_id
    ) != set(transformation_ids):
        raise ValueError("candidate transformation artifacts do not match lifecycle identifiers")
    ordered_transformations = [
        transformations_by_id[transformation_id] for transformation_id in transformation_ids
    ]
    trusted_by_id = {
        transformation.transformation_id: transformation
        for transformation in trusted_transformations
    }
    if (
        len(trusted_by_id) != len(trusted_transformations)
        or tuple(transformation.transformation_id for transformation in trusted_transformations)
        != transformation_ids
    ):
        raise ValueError("trusted lowering did not reproduce the accepted transformation sequence")
    for _path, transformation in ordered_transformations:
        trusted = trusted_by_id.get(transformation.transformation_id)
        if trusted is None or canonical_json(transformation) != canonical_json(trusted):
            raise ValueError(
                "persisted transformation does not match the trusted lowering derivation"
            )
    expected_source_hash = baseline_genome_hash
    previous_transformation_id: str | None = None
    for _path, transformation in ordered_transformations:
        delta = transformation.extensions.root.get("sloforge.dev/applied-delta")
        if not isinstance(delta, dict):
            raise ValueError("transformation is missing its canonical applied delta")
        source_hash = delta.get("source_genome_hash")
        target_hash = delta.get("target_genome_hash")
        if not isinstance(source_hash, str) or not isinstance(target_hash, str):
            raise ValueError("transformation applied delta is missing source or target hashes")
        source_constraint = f"source_genome_sha256 == {source_hash}"
        target_constraint = f"target_genome_sha256 == {target_hash}"
        if source_hash != expected_source_hash:
            raise ValueError("transformation source does not continue the derivation chain")
        if source_constraint not in transformation.source_pattern.structural_constraints:
            raise ValueError("transformation source pattern does not match its applied delta")
        if target_constraint not in transformation.target_pattern.structural_constraints:
            raise ValueError("transformation target pattern does not match its applied delta")
        expected_parents = (
            () if previous_transformation_id is None else (previous_transformation_id,)
        )
        if transformation.parent_transformations != expected_parents:
            raise ValueError("transformation parent does not match the derivation chain")
        if not transformation.verification_obligations:
            raise ValueError("transformation is missing verification obligations")
        expected_source_hash = target_hash
        previous_transformation_id = transformation.transformation_id
    if expected_source_hash != candidate_genome_hash:
        raise ValueError("transformation derivation does not terminate at the accepted genome")
    return ordered_transformations


def build_local_capsule(
    candidate_directory: Path,
    output_directory: Path,
    *,
    observed_at: datetime,
    trust_output: Path | None = None,
) -> CapsuleBuildResult:
    """Build and immediately independently validate a local CPU/simulation capsule."""

    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise ValueError("observed_at must be timezone-aware")
    observed_at = observed_at.astimezone(UTC)
    if output_directory.exists() and output_directory.is_symlink():
        raise ValueError("capsule output directory must not be a symlink")
    context_path = trust_output or output_directory.with_name(
        f"{output_directory.name}.validation-context.json"
    )
    if context_path.exists() and context_path.is_symlink():
        raise ValueError("capsule trust output must not be a symlink")
    if context_path.resolve().is_relative_to(output_directory.resolve()):
        raise ValueError("capsule trust output must be outside the untrusted capsule directory")
    if context_path.exists():
        raise FileExistsError(f"refusing to overwrite capsule trust output: {context_path}")
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"capsule output directory is not empty: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)
    run_directory = candidate_directory.parent.parent
    repository = Path(__file__).resolve().parents[4]
    candidate = load_candidate(candidate_directory / "candidate.json")
    if candidate.state is not CandidateSuccessState.SIMULATED:
        raise ValueError(
            "only a model-checked and simulated candidate can enter capsule construction"
        )
    design = CandidateDesign.model_validate_json(
        (candidate_directory / "candidate_design.json").read_bytes(), strict=True
    )
    protocol = CancellationPolicyVerifier().verify(design, None, seed=design.seed)
    if not protocol.passed:
        raise ValueError("independent cancellation verification failed during capsule build")
    manifest = _read_object(run_directory / "run_manifest.json")
    synthesis = _read_object(run_directory / "synthesis/result.json")
    if synthesis.get("accepted_candidate_id") != candidate.candidate_id:
        raise ValueError("candidate is not the accepted result of this synthesis run")
    if synthesis.get("runtime_differential_passed") is not True:
        raise ValueError("runtime differential evidence did not pass")
    workload_record = manifest.get("workload_contract")
    hardware_record = manifest.get("hardware_contract")
    if not isinstance(workload_record, dict) or not isinstance(hardware_record, dict):
        raise ValueError("run manifest is missing workload or hardware provenance")
    workload_path = Path(str(workload_record["path"]))
    hardware_path = Path(str(hardware_record["path"]))
    workload_digest = Digest(value=str(workload_record["sha256"]))
    hardware_digest = Digest(value=str(hardware_record["sha256"]))
    if _digest(workload_path.read_bytes()) != workload_digest:
        raise ValueError("workload changed after synthesis")
    if _digest(hardware_path.read_bytes()) != hardware_digest:
        raise ValueError("hardware contract changed after synthesis")
    hardware_document = _read_object(hardware_path)
    architecture = str(hardware_document.get("architecture", "cpu"))
    measured_fingerprint = hardware_document.get("measured_fingerprint")
    if not isinstance(measured_fingerprint, str):
        raise ValueError("hardware contract requires a measured_fingerprint distinct from its hash")
    hardware_fingerprint = Digest(value=measured_fingerprint)
    baseline_runtime_config = _read_object(run_directory / "generated_runtime/runtime_config.json")
    package_root = Path(str(baseline_runtime_config["reference_package_root"]))
    package = load_reference_package(package_root)
    persisted_package_hashes = {
        str(manifest.get("package_hash")),
        str(baseline_runtime_config.get("package_hash")),
    }
    if persisted_package_hashes != {package.package_hash}:
        raise ValueError(
            "reference package identity changed after inspection or runtime generation"
        )
    package_manifest = _read_object(package_root / "reference_package.json")
    tokenizer_path = package_root / str(package_manifest["tokenizer_module"])
    tokenizer_digest = _digest(tokenizer_path.read_bytes())
    source_digest = Digest(value=package.package_hash)
    genome_document = json.loads(
        (candidate_directory / "inference_genome.json").read_text(encoding="utf-8")
    )
    if candidate.genome_hash.value != canonical_hash(genome_document):
        raise ValueError("candidate genome changed after acceptance")

    baseline = load_inference_genome(run_directory / "inference_genome.json")
    if canonical_hash(baseline) != str(synthesis["baseline_genome_hash"]):
        raise ValueError("synthesis baseline genome does not match the persisted baseline")
    constraint_document = ConstraintDocument.model_validate_json(
        (run_directory / "synthesis/cegis/constraints.json").read_bytes(), strict=True
    )
    cegis_result = CegisRunResult.model_validate_json(
        (run_directory / "synthesis/cegis_result.json").read_bytes(), strict=True
    )
    trusted_lowering = lower_candidate(
        baseline,
        design,
        learned_constraints=tuple(
            constraint.learned for constraint in constraint_document.constraints
        ),
        counterexample_references=cegis_result.counterexample_ids,
    )
    if canonical_json(trusted_lowering.design) != canonical_json(design):
        raise ValueError("candidate design does not match the trusted lowering result")
    if canonical_hash(trusted_lowering.genome) != candidate.genome_hash.value:
        raise ValueError("candidate genome does not match the trusted lowering result")
    if canonical_json(trusted_lowering.genome) != canonical_json(genome_document):
        raise ValueError("persisted candidate genome differs from the trusted lowering result")

    transformation_paths = sorted((candidate_directory / "transformations").glob("*.json"))
    ordered_transformations = _validate_transformation_chain(
        transformation_paths,
        transformation_ids=candidate.transformation_ids,
        baseline_genome_hash=str(synthesis["baseline_genome_hash"]),
        candidate_genome_hash=candidate.genome_hash.value,
        trusted_transformations=trusted_lowering.transformations,
    )
    transformation_paths = [path for path, _transformation in ordered_transformations]

    artifacts: list[ArtifactRef] = []
    runtime_bundle = _runtime_bundle_bytes(
        run_directory,
        candidate_directory,
        package_root,
        candidate_id=candidate.candidate_id,
        candidate_genome_hash=candidate.genome_hash.value,
    )
    artifacts.append(
        _artifact(
            output_directory,
            "generated-runtime",
            ArtifactRole.GENERATED_RUNTIME,
            runtime_bundle,
            origin=ArtifactOrigin.GENERATED_UNTRUSTED,
            suffix=".zip",
            media_type="application/zip",
        )
    )
    for index, transformation_path in enumerate(transformation_paths):
        artifacts.append(
            _copy_artifact(
                output_directory,
                f"transformation-{index:03d}",
                ArtifactRole.SEMANTIC_EVIDENCE,
                transformation_path,
                origin=ArtifactOrigin.VERIFIED_EVIDENCE,
                suffix=".json",
                media_type="application/json",
            )
        )
    artifacts.append(
        _copy_artifact(
            output_directory,
            "generated-policy",
            ArtifactRole.GENERATED_POLICY,
            candidate_directory / "policy.slo",
            origin=ArtifactOrigin.GENERATED_UNTRUSTED,
            suffix=".slo",
            media_type="text/plain",
        )
    )
    artifacts.append(
        _copy_artifact(
            output_directory,
            "generated-policy-bytecode",
            ArtifactRole.GENERATED_POLICY,
            candidate_directory / "policy.bytecode.json",
            origin=ArtifactOrigin.GENERATED_UNTRUSTED,
            suffix=".json",
            media_type="application/json",
        )
    )
    deployment_payload = canonical_json(
        {
            "schema_version": "1.0.0",
            "mode": "local-shadow-first",
            "candidate_id": candidate.candidate_id,
            "external_live_promotion": False,
            "transition": "request-boundary",
        }
    )
    artifacts.append(
        _artifact(
            output_directory,
            "deployment",
            ArtifactRole.DEPLOYMENT,
            deployment_payload,
            origin=ArtifactOrigin.GENERATED_UNTRUSTED,
        )
    )
    rollback_payload = canonical_json(
        {
            "schema_version": "1.0.0",
            "action": "restore-previous-champion",
            "preserve_active_streams": True,
            "external_execution": False,
        }
    )
    rollback = _artifact(
        output_directory,
        "rollback",
        ArtifactRole.ROLLBACK,
        rollback_payload,
        origin=ArtifactOrigin.TRUSTED,
    )
    artifacts.append(rollback)
    lock = _copy_artifact(
        output_directory,
        "dependency-lock",
        ArtifactRole.DEPENDENCY_LOCK,
        repository / "uv.lock",
        origin=ArtifactOrigin.TRUSTED,
        suffix=".lock",
        media_type="text/plain",
    )
    artifacts.append(lock)

    semantic_payload = canonical_json(
        {
            "schema_version": "1.0.0",
            "candidate_id": candidate.candidate_id,
            "candidate_genome_hash": candidate.genome_hash.value,
            "runtime_differential_passed": True,
            "sandbox_termination": synthesis["sandbox_termination"],
            "protocol_evidence_id": protocol.evidence_id,
            "seed": design.seed,
        }
    )
    semantic = _artifact(
        output_directory,
        "semantic-evidence",
        ArtifactRole.SEMANTIC_EVIDENCE,
        semantic_payload,
        origin=ArtifactOrigin.VERIFIED_EVIDENCE,
    )
    artifacts.append(semantic)
    differential_path = candidate_directory / "evidence/runtime-differential-result.json"
    differential = _read_object(differential_path)
    policy_digest = hashlib.sha256(
        (candidate_directory / "policy.bytecode.json").read_bytes()
    ).hexdigest()
    if differential.get("candidate_id") != candidate.candidate_id:
        raise ValueError("differential evidence candidate identity mismatch")
    if differential.get("candidate_genome_hash") != candidate.genome_hash.value:
        raise ValueError("differential evidence genome identity mismatch")
    if differential.get("policy_bytecode_sha256") != policy_digest:
        raise ValueError("differential evidence policy identity mismatch")
    runtime_hashes = differential.get("runtime_artifact_hashes")
    candidate_runtime = candidate_directory / "generated_runtime"
    if not isinstance(runtime_hashes, dict) or any(
        not isinstance(name, str)
        or not isinstance(digest, str)
        or not (candidate_runtime / name).is_file()
        or hashlib.sha256((candidate_runtime / name).read_bytes()).hexdigest() != digest
        for name, digest in runtime_hashes.items()
    ):
        raise ValueError("differential evidence runtime artifact identities do not match")
    final_corpus = package_root / str(
        package_manifest["quality_contract"]["final_evaluation_corpus"]
    )
    if differential.get("corpus_sha256") != hashlib.sha256(final_corpus.read_bytes()).hexdigest():
        raise ValueError("differential evidence corpus digest mismatch")
    cases = differential.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("differential evidence has no replayable cases")
    runtime_config = _read_object(candidate_runtime / "runtime_config.json")
    if (
        runtime_config.get("package_hash") != package.package_hash
        or Path(str(runtime_config.get("reference_package_root"))).resolve()
        != package_root.resolve()
    ):
        raise ValueError("candidate runtime is bound to another reference package")
    replay = _independent_runtime_differential(
        candidate_runtime,
        package_root,
        final_corpus,
        seed=int(runtime_config["generation_seed"]),
    )
    if replay.get("cases") != cases or replay.get("failures", []) != differential.get(
        "failures", []
    ):
        raise ValueError("differential evidence does not match independent sandbox replay")
    corpus_cases = [
        json.loads(line)
        for line in final_corpus.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(corpus_cases) != len(cases):
        raise ValueError("differential evidence does not cover the complete final corpus")
    for ordinal, (case, corpus_case) in enumerate(zip(cases, corpus_cases, strict=True), start=1):
        if (
            not isinstance(case, dict)
            or not isinstance(corpus_case, dict)
            or case.get("line") != ordinal
            or case.get("request_seed") != corpus_case.get("seed")
            or case.get("expected") != corpus_case.get("expected_tokens")
            or case.get("observed") != corpus_case.get("expected_tokens")
            or case.get("exact_match") is not True
        ):
            raise ValueError("differential case is not bound to its final-corpus oracle")
    exact_count = sum(isinstance(case, dict) and case.get("exact_match") is True for case in cases)
    observed_quality = exact_count / len(cases)
    if differential.get("passed") is not True or observed_quality != 1.0:
        raise ValueError("differential quality evidence did not satisfy exact match")
    quality_payload = canonical_json(
        {
            **differential,
            "metric": "exact_token_match",
            "threshold": 1.0,
            "observed": observed_quality,
            "case_count": len(cases),
            "search_data_separate": True,
        }
    )
    quality = _artifact(
        output_directory,
        "quality-evidence",
        ArtifactRole.QUALITY_EVIDENCE,
        quality_payload,
        origin=ArtifactOrigin.VERIFIED_EVIDENCE,
    )
    artifacts.append(quality)
    memory_capacity = int(hardware_document.get("memory_bytes", 8 * 1024**3))
    resource_document = _resource_evidence(
        runtime_config,
        genome_document,
        package_manifest,
        capacity_bytes=memory_capacity,
        bundle_size_bytes=len(runtime_bundle),
    )
    resource_payload = canonical_json(resource_document)
    if resource_document["passed"] is not True:
        raise ValueError("conservative resource analysis rejected the candidate")
    resource = _artifact(
        output_directory,
        "resource-evidence",
        ArtifactRole.RESOURCE_EVIDENCE,
        resource_payload,
        origin=ArtifactOrigin.VERIFIED_EVIDENCE,
    )
    artifacts.append(resource)
    modelcheck_path = candidate_directory / "evidence/modelcheck-result.json"
    modelcheck_document = _read_object(modelcheck_path)
    expected_modelcheck = bounded_candidate_modelcheck_document(design, seed=int(synthesis["seed"]))
    if canonical_json(modelcheck_document) != canonical_json(expected_modelcheck):
        raise ValueError(
            "model-check evidence does not match the independently recomputed bounded result"
        )
    if modelcheck_document.get("result") != "pass":
        raise ValueError("candidate bounded model-check result is not passing")
    operational = _copy_artifact(
        output_directory,
        "operational-evidence",
        ArtifactRole.MODEL_CHECK_RESULT,
        modelcheck_path,
        origin=ArtifactOrigin.FORMAL_OR_BOUNDED_EVIDENCE,
        suffix=".json",
        media_type="application/json",
    )
    artifacts.append(operational)
    property_path = candidate_directory / "evidence/property-result.json"
    property_document = _read_object(property_path)
    expected_property = bounded_candidate_policy_property_document(
        design, seed=int(synthesis["seed"])
    )
    if canonical_json(property_document) != canonical_json(expected_property):
        raise ValueError(
            "policy property evidence does not match the independently recomputed result"
        )
    if property_document.get("result") != "pass":
        raise ValueError("candidate bounded property result is not passing")
    property_artifact = _copy_artifact(
        output_directory,
        "policy-property-evidence",
        ArtifactRole.PROPERTY_TEST_RESULT,
        property_path,
        origin=ArtifactOrigin.FORMAL_OR_BOUNDED_EVIDENCE,
        suffix=".json",
        media_type="application/json",
    )
    artifacts.append(property_artifact)
    simulation_path = candidate_directory / "evidence/simulation-result.json"
    simulation_document = _read_object(simulation_path)
    runtime_manifest_path = candidate_runtime / "candidate_runtime_manifest.json"
    if (
        simulation_document.get("candidate_id") != candidate.candidate_id
        or simulation_document.get("candidate_genome_hash") != candidate.genome_hash.value
        or simulation_document.get("policy_bytecode_sha256") != policy_digest
        or simulation_document.get("runtime_manifest_sha256")
        != _digest(runtime_manifest_path.read_bytes()).value
        or simulation_document.get("result") != "pass"
        or simulation_document.get("comparison_permitted") is not False
        or simulation_document.get("workload_sha256") != workload_digest.value
    ):
        raise ValueError("candidate simulation evidence is invalid or misbound")
    simulation_artifact = _copy_artifact(
        output_directory,
        "candidate-simulation",
        ArtifactRole.PERFORMANCE_SAMPLES,
        simulation_path,
        origin=ArtifactOrigin.PERFORMANCE_EVIDENCE,
        suffix=".json",
        media_type="application/json",
    )
    artifacts.append(simulation_artifact)

    counterexample_refs: list[str] = []
    counterexample_directory = run_directory / "synthesis/cegis/counterexamples"
    for index, source in enumerate(sorted(counterexample_directory.glob("*.json"))):
        artifact_id = f"counterexample-{index:03d}"
        artifacts.append(
            _copy_artifact(
                output_directory,
                artifact_id,
                ArtifactRole.SEMANTIC_EVIDENCE,
                source,
                origin=ArtifactOrigin.VERIFIED_EVIDENCE,
                suffix=".json",
                media_type="application/json",
            )
        )
        counterexample_refs.append(artifact_id)
    corpus_document = CounterexampleCorpus(
        candidate_genome_hash=Digest(value=candidate.genome_hash.value),
        counterexample_artifact_ids=tuple(counterexample_refs),
        searched_domains=(
            "deadline batching cancellation schedules bounded to six events",
            "generated runtime final evaluation corpus",
        ),
    )
    corpus = _artifact(
        output_directory,
        "counterexample-corpus",
        ArtifactRole.COUNTEREXAMPLE_CORPUS,
        corpus_document.model_dump_json().encode(),
        origin=ArtifactOrigin.VERIFIED_EVIDENCE,
    )
    artifacts.append(corpus)

    evidence_artifacts = {
        EvidenceClass.SEMANTIC: semantic,
        EvidenceClass.QUALITY: quality,
        EvidenceClass.RESOURCE: resource,
        EvidenceClass.PERFORMANCE: simulation_artifact,
        EvidenceClass.OPERATIONAL: operational,
        EvidenceClass.PROPERTY_TEST: property_artifact,
    }
    issuers = {
        EvidenceClass.SEMANTIC: EvidenceIssuer.OPERATOR_VERIFIER,
        EvidenceClass.QUALITY: EvidenceIssuer.QUALITY_HARNESS,
        EvidenceClass.RESOURCE: EvidenceIssuer.RESOURCE_ANALYZER,
        EvidenceClass.PERFORMANCE: EvidenceIssuer.BENCHMARK_HARNESS,
        EvidenceClass.OPERATIONAL: EvidenceIssuer.MODEL_CHECKER,
        EvidenceClass.PROPERTY_TEST: EvidenceIssuer.PROPERTY_HARNESS,
    }
    levels = {
        EvidenceClass.SEMANTIC: VerificationLevel.DIFFERENTIAL,
        EvidenceClass.QUALITY: VerificationLevel.DIFFERENTIAL,
        EvidenceClass.RESOURCE: VerificationLevel.PROPERTY,
        EvidenceClass.PERFORMANCE: VerificationLevel.PROPERTY,
        EvidenceClass.OPERATIONAL: VerificationLevel.BOUNDED_EXHAUSTIVE,
        EvidenceClass.PROPERTY_TEST: VerificationLevel.BOUNDED_EXHAUSTIVE,
    }
    valid_until = observed_at + timedelta(days=30)
    evidence = tuple(
        EvidenceRecord(
            evidence_id=f"evidence:{evidence_class.value}",
            evidence_class=evidence_class,
            level=levels[evidence_class],
            result=EvidenceResult.PASS,
            issuer=issuers[evidence_class],
            issuer_version="1.0.0",
            artifact_ids=(
                (artifact.artifact_id, rollback.artifact_id)
                if evidence_class is EvidenceClass.OPERATIONAL
                else (artifact.artifact_id,)
            ),
            observed_at=observed_at,
            valid_until=valid_until,
            deterministic_seed=design.seed,
            assumptions=(
                ("deterministic simulation only; no hardware timing claim")
                if evidence_class is EvidenceClass.PERFORMANCE
                else "evidence is scoped to the declared HybridDecoder input domain",
            ),
        )
        for evidence_class, artifact in evidence_artifacts.items()
    )
    statement = {
        ClaimCategory.SEMANTIC: "generated runtime matches the reference corpus and cancellation policy",
        ClaimCategory.QUALITY: "generated runtime achieves exact token match on the final local corpus",
        ClaimCategory.RESOURCE: "declared local champion/challenger coexistence fits the CPU memory contract",
        ClaimCategory.PERFORMANCE: "candidate-bound deterministic simulation completed; no performance improvement is accepted",
        ClaimCategory.OPERATIONAL: "candidate cancellation policy passes the declared bounded event schedule",
    }
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
            statement=statement[category],
            scope=ClaimScope(
                input_domain=("HybridDecoder contract: batch=1, sequence=1..64, generated=1..16",),
                shape_domain=("input_ids[1,1..64]",),
                dtype_domain=("int64 input and declared persistent-state dtypes",),
                assumptions=(
                    "performance is a deterministic simulator result, not hardware evidence",
                )
                if category is ClaimCategory.PERFORMANCE
                else (),
                exclusions=("unseen custom operators outside the package contract",),
            ),
            level=levels[evidence_class],
            result=EvidenceResult.PASS,
            evidence_ids=(f"evidence:{evidence_class.value}",),
            promotion_required=False,
        )
        for category, evidence_class in category_class.items()
    )
    dependency = DependencyRequirement(name="sloforge", version="0.1.0")
    identity = CapsuleIdentity(
        candidate_genome_hash=Digest(value=candidate.genome_hash.value),
        source_model_hash=source_digest,
        tokenizer_hash=tokenizer_digest,
        workload_contract_hash=workload_digest,
        hardware_contract_hash=hardware_digest,
        compiler_version="0.1.0",
        verifier_version="1.0.0",
        git_commit=_git_commit(repository),
        dependency_lock_hash=lock.digest,
        generated_at=observed_at,
    )
    capsule = seal_capsule(
        GenesisCapsule(
            identity=identity,
            artifacts=tuple(artifacts),
            dependencies=(dependency,),
            hardware=HardwareCompatibility(
                hardware_contract_hash=hardware_digest,
                allowed_fingerprints=(hardware_fingerprint,),
                architectures=(architecture,),
                restrictions=("CPU-only local evidence; CUDA and multi-node paths unexercised",),
            ),
            evidence=evidence,
            claims=claims,
            benchmarks=(),
            known_unsupported_cases=(
                "hardware performance is not established by this local capsule",
                "multi-node state transfer is not exercised",
            ),
            unverified_assumptions=(
                "policy performance effects require independent repeated hardware or service benchmarking",
            ),
        )
    )
    if capsule.capsule_digest is None:  # defensive: seal_capsule must always address the manifest
        raise RuntimeError("sealed capsule has no digest")
    context = ValidationContext(
        expected_capsule_digest=capsule.capsule_digest,
        source_model_hash=source_digest,
        tokenizer_hash=tokenizer_digest,
        workload_contract_hash=workload_digest,
        hardware_contract_hash=hardware_digest,
        hardware_fingerprint=hardware_fingerprint,
        hardware_architecture=architecture,
        device_count=1,
        dependency_lock_hash=lock.digest,
        dependencies=(CurrentDependency(name="sloforge", version="0.1.0"),),
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
                            item.digest for item in artifacts if item.artifact_id == artifact_id
                        ),
                    )
                    for artifact_id in record.artifact_ids
                ),
            )
            for record in evidence
        ),
        trusted_artifact_anchors=tuple(
            TrustedArtifactAnchor(artifact_id=item.artifact_id, digest=item.digest)
            for item in artifacts
            if item.origin is ArtifactOrigin.TRUSTED
        ),
        trusted_verifier_version="1.0.0",
        now=observed_at,
        require_promotion_evidence=False,
    )
    report = validate_capsule(capsule, output_directory, context)
    if not report.local_evolution_eligible:
        issues = "; ".join(f"{item.code.value}:{item.path}" for item in report.issues)
        raise ValueError(f"newly built capsule failed independent validation: {issues}")
    manifest_path = publish_capsule(capsule, output_directory / "manifests")
    _write_once(context_path, canonical_json(context) + b"\n")
    return CapsuleBuildResult(
        capsule_path=str(manifest_path.resolve()),
        capsule_digest=capsule.capsule_digest.value,
        context_path=str(context_path.resolve()),
        candidate_id=candidate.candidate_id,
        artifact_count=len(capsule.artifacts),
        evidence_count=len(capsule.evidence),
        claim_count=len(capsule.claims),
        promotion_eligible=False,
        local_evolution_eligible=True,
        external_production_eligible=False,
        performance_scope="candidate-bound deterministic simulation; no improvement claim",
        hardware_backed=False,
    )


__all__ = ["CapsuleBuildResult", "build_local_capsule"]
