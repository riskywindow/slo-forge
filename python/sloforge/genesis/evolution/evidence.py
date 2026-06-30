"""Artifact-backed local shadow and canary evidence from capsule runtimes."""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..capsule import ArtifactRole, VerificationLevel, load_capsule
from ..ir import canonical_json
from ..sandbox import (
    IsolationStatus,
    SandboxCapabilities,
    SandboxLimits,
    SandboxRequest,
    execute_sandboxed,
)
from .models import CapsuleReference, ChallengerSpec, GateObservation, GateStage


class _ObservationCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    request_id: str
    request_seed: int
    token_ids: tuple[int, ...]
    token_count: int = Field(ge=0)
    ttft_ns: int = Field(gt=0)
    mean_tpot_ns: int = Field(gt=0)
    completion_ns: int = Field(gt=0)
    error: str | None


class _RuntimeObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    schema_version: Literal["sloforge.genesis.evolution.runtime-observation/v1"]
    seed: int = Field(ge=0)
    request_count: Annotated[int, Field(ge=1, le=256)]
    cases: tuple[_ObservationCase, ...]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"runtime quality corpus {field} must be an integer")
    return value


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite evolution evidence: {path}")
    path.write_bytes(payload)


def _artifact_path(capsule_path: Path, relative: str) -> Path:
    manifest_parent = capsule_path.parent.resolve(strict=True)
    root = manifest_parent.parent if manifest_parent.name == "manifests" else manifest_parent
    candidate = root.joinpath(*relative.split("/"))
    if candidate.is_symlink():
        raise ValueError("capsule runtime artifact must not be a symlink")
    resolved = candidate.resolve(strict=True)
    resolved.relative_to(root)
    if not resolved.is_file():
        raise ValueError("capsule runtime artifact must be a regular file")
    return resolved


def _extract_runtime(reference: CapsuleReference, destination: Path) -> tuple[Path, str]:
    manifest_path = Path(reference.path)
    capsule = load_capsule(manifest_path)
    if (
        capsule.capsule_digest is None
        or capsule.capsule_digest.value != reference.capsule_digest
        or capsule.identity.candidate_genome_hash.value != reference.genome_hash
    ):
        raise ValueError("capsule manifest does not match the gate-evidence reference")
    runtime_refs = [
        item for item in capsule.artifacts if item.role is ArtifactRole.GENERATED_RUNTIME
    ]
    if len(runtime_refs) != 1:
        raise ValueError("capsule must contain exactly one generated runtime bundle")
    runtime_ref = runtime_refs[0]
    archive_path = _artifact_path(manifest_path, runtime_ref.path)
    payload = archive_path.read_bytes()
    if _sha256(payload) != runtime_ref.digest.value:
        raise ValueError("generated runtime bundle changed after capsule validation")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"runtime extraction destination already exists: {destination}")
    destination.mkdir(parents=True)
    total_size = 0
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        if not 1 <= len(members) <= 4096:
            raise ValueError("runtime bundle entry count is outside the trusted bound")
        names = [member.filename for member in members]
        if len(names) != len(set(names)):
            raise ValueError("runtime bundle contains duplicate paths")
        for member in members:
            path = Path(member.filename)
            mode = member.external_attr >> 16
            if path.is_absolute() or ".." in path.parts or stat.S_ISLNK(mode) or member.is_dir():
                raise ValueError("runtime bundle contains an unsafe entry")
            total_size += member.file_size
            if total_size > 64 * 1024 * 1024:
                raise ValueError("runtime bundle exceeds the trusted extraction bound")
            target = destination.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))
            target.chmod(0o444)
    return destination, runtime_ref.digest.value


def _trace_from_bundle(bundle: Path, *, count: int, seed: int, stage: GateStage) -> bytes:
    manifest = json.loads(
        (bundle / "reference_package/reference_package.json").read_text(encoding="utf-8")
    )
    corpus_name = manifest["quality_contract"]["final_evaluation_corpus"]
    corpus_path = (bundle / "reference_package" / str(corpus_name)).resolve(strict=True)
    corpus_path.relative_to((bundle / "reference_package").resolve(strict=True))
    source: list[dict[str, object]] = []
    for line in corpus_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError("runtime quality corpus row must be an object")
            source.append(item)
    if not source:
        raise ValueError("runtime quality corpus is empty")
    requests = []
    for index in range(count):
        sample = source[index % len(source)]
        requests.append(
            {
                "request_id": f"{stage.value}-{index:04d}",
                "text": str(sample["text"]),
                "maximum_new_tokens": _integer(
                    sample["maximum_new_tokens"], field="maximum_new_tokens"
                ),
                "seed": _integer(sample.get("seed", seed + index), field="seed"),
                "batching_eligible": True,
            }
        )
    return (
        canonical_json(
            {
                "schema_version": "sloforge.genesis.evolution.trace/v1",
                "seed": seed,
                "stage": stage.value,
                "requests": requests,
            }
        )
        + b"\n"
    )


def _execute_runtime(
    bundle: Path,
    trace_path: Path,
    output: Path,
    *,
    seed: int,
) -> tuple[_RuntimeObservation, dict[str, object]]:
    runner = Path(__file__).with_name("runtime_evidence_runner.py").resolve(strict=True)
    repository_python = Path(__file__).resolve().parents[3]
    result_path = output / "runtime-observation.json"
    result = execute_sandboxed(
        SandboxRequest(
            argv=(
                sys.executable,
                str(runner),
                "--bundle",
                str(bundle),
                "--trace",
                str(trace_path),
                "--output",
                str(result_path),
                "--seed",
                str(seed),
                "--timeout-seconds",
                "3",
            ),
            working_directory=bundle,
            read_only_paths=(bundle, trace_path, repository_python, Path(sys.prefix)),
            artifact_output_directory=output,
            seed=seed,
            limits=SandboxLimits(
                wall_time_seconds=20.0,
                cpu_time_seconds=15,
                memory_bytes=2 * 1024 * 1024 * 1024,
                process_count=1,
                output_bytes=64 * 1024,
                artifact_bytes=4 * 1024 * 1024,
                artifact_entries=32,
                open_files=64,
            ),
        )
    )
    if not result.succeeded or not result_path.is_file():
        raise RuntimeError(
            f"sandboxed capsule runtime evidence failed: {result.termination.value}: {result.stderr}"
        )
    observation = _RuntimeObservation.model_validate_json(result_path.read_bytes(), strict=True)
    if observation.seed != seed or observation.request_count != len(observation.cases):
        raise RuntimeError("runtime observation is inconsistent with its trusted invocation")
    sandbox: dict[str, object] = {
        "termination": result.termination.value,
        "return_code": result.return_code,
        "duration_seconds": result.duration_seconds,
        "output_bytes": result.output_bytes,
        "process_group_cleaned": result.process_group_cleaned,
        "capabilities": result.capabilities.model_dump(mode="json"),
        "stderr": result.stderr,
    }
    return observation, sandbox


def _percentile(values: list[int], quantile: float) -> int:
    if not values:
        raise ValueError("cannot calculate a percentile of no samples")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * quantile + 0.5)))
    return ordered[index]


def collect_local_gate_evidence(
    *,
    champion: CapsuleReference,
    challenger: CapsuleReference,
    candidate_id: str,
    stage: GateStage,
    sample_count: int,
    seed: int,
    observed_at_ms: int,
    output_directory: Path,
) -> GateObservation:
    """Run both validated capsule bundles on one trace and hash all raw evidence."""

    if not 1 <= sample_count <= 256:
        raise ValueError("local gate sample_count must be in [1, 256]")
    if output_directory.exists() or output_directory.is_symlink():
        raise FileExistsError(f"gate evidence output already exists: {output_directory}")
    output_directory.mkdir(parents=True)
    extraction = output_directory / "extracted"
    champion_bundle, champion_bundle_digest = _extract_runtime(champion, extraction / "champion")
    challenger_bundle, challenger_bundle_digest = _extract_runtime(
        challenger, extraction / "challenger"
    )
    trace_payload = _trace_from_bundle(champion_bundle, count=sample_count, seed=seed, stage=stage)
    trace_path = output_directory / "trace.json"
    _write_once(trace_path, trace_payload)
    champion_observation, champion_sandbox = _execute_runtime(
        champion_bundle, trace_path, output_directory / "champion", seed=seed
    )
    challenger_observation, challenger_sandbox = _execute_runtime(
        challenger_bundle, trace_path, output_directory / "challenger", seed=seed
    )
    champion_cases = {item.request_id: item for item in champion_observation.cases}
    challenger_cases = {item.request_id: item for item in challenger_observation.cases}
    if set(champion_cases) != set(challenger_cases) or len(challenger_cases) != sample_count:
        raise RuntimeError("champion and challenger did not execute the identical bounded trace")
    mismatches = []
    errors = 0
    interrupted = 0
    for request_id in sorted(champion_cases):
        expected = champion_cases[request_id]
        observed = challenger_cases[request_id]
        if expected.error is not None or observed.error is not None:
            errors += 1
            interrupted += 1
        if expected.token_ids != observed.token_ids:
            mismatches.append(
                {
                    "request_id": request_id,
                    "champion_token_ids": list(expected.token_ids),
                    "challenger_token_ids": list(observed.token_ids),
                }
            )
    champion_ttft = _percentile([item.ttft_ns for item in champion_cases.values()], 0.95)
    challenger_ttft = _percentile([item.ttft_ns for item in challenger_cases.values()], 0.95)
    champion_tpot = _percentile([item.mean_tpot_ns for item in champion_cases.values()], 0.99)
    challenger_tpot = _percentile([item.mean_tpot_ns for item in challenger_cases.values()], 0.99)
    artifact = {
        "schema_version": "sloforge.genesis.evolution.gate-evidence/v1",
        "stage": stage.value,
        "seed": seed,
        "candidate_id": candidate_id,
        "champion_capsule_digest": champion.capsule_digest,
        "challenger_capsule_digest": challenger.capsule_digest,
        "champion_runtime_bundle_digest": champion_bundle_digest,
        "challenger_runtime_bundle_digest": challenger_bundle_digest,
        "trace_sha256": _sha256(trace_payload),
        "trace_request_count": sample_count,
        "champion_observation": champion_observation.model_dump(mode="json"),
        "challenger_observation": challenger_observation.model_dump(mode="json"),
        "champion_sandbox": champion_sandbox,
        "challenger_sandbox": challenger_sandbox,
        "comparison": {
            "mismatches": mismatches,
            "error_count": errors,
            "interrupted_streams": interrupted,
            "champion_p95_ttft_ns": champion_ttft,
            "challenger_p95_ttft_ns": challenger_ttft,
            "champion_p99_mean_tpot_ns": champion_tpot,
            "challenger_p99_mean_tpot_ns": challenger_tpot,
        },
        "scope": "bounded deterministic local replay; timing is host wall-clock evidence only",
        "hardware_backed": False,
    }
    artifact_payload = canonical_json(artifact) + b"\n"
    artifact_path = output_directory / "gate-evidence.json"
    _write_once(artifact_path, artifact_payload)
    shutil.rmtree(extraction)
    return GateObservation(
        event_id=f"fixture-{stage.value}-evidence",
        candidate_id=candidate_id,
        capsule_digest=challenger.capsule_digest,
        evidence_digest=_sha256(artifact_payload),
        stage=stage,
        verification_level=VerificationLevel.PROPERTY,
        observed_at_ms=observed_at_ms,
        deterministic_seed=seed,
        sample_count=sample_count,
        error_rate=errors / sample_count,
        p95_ttft_ratio=challenger_ttft / champion_ttft,
        p99_tpot_ratio=challenger_tpot / champion_tpot,
        quality_regression=len(mismatches) / sample_count,
        interrupted_streams=interrupted,
    )


def local_gate_evidence_validator(
    evidence_root: Path,
) -> Callable[[GateObservation, ChallengerSpec], bool]:
    """Return a fail-closed validator for artifacts emitted by local gate replay.

    The callback independently derives every gate metric from the bounded raw
    runtime observations. The summary embedded in the evidence file is checked,
    never trusted as the source of the controller decision.
    """

    root = evidence_root.resolve()

    def validate(observation: GateObservation, challenger: ChallengerSpec) -> bool:
        try:
            candidate_id = challenger.candidate_id
            capsule_digest = challenger.capsule.capsule_digest
            path = root / observation.stage.value / "gate-evidence.json"
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
                return False
            payload = path.read_bytes()
            if _sha256(payload) != observation.evidence_digest:
                return False
            document = json.loads(payload)
            if not isinstance(document, dict):
                return False
            comparison = document.get("comparison")
            champion_document = document.get("champion_observation")
            candidate_document = document.get("challenger_observation")
            champion_sandbox = document.get("champion_sandbox")
            challenger_sandbox = document.get("challenger_sandbox")
            if not all(
                isinstance(item, dict)
                for item in (
                    comparison,
                    champion_document,
                    candidate_document,
                    champion_sandbox,
                    challenger_sandbox,
                )
            ):
                return False
            assert isinstance(comparison, dict)
            assert isinstance(champion_document, dict)
            assert isinstance(candidate_document, dict)
            assert isinstance(champion_sandbox, dict)
            assert isinstance(challenger_sandbox, dict)
            champion = _RuntimeObservation.model_validate_json(
                canonical_json(champion_document), strict=True
            )
            candidate = _RuntimeObservation.model_validate_json(
                canonical_json(candidate_document), strict=True
            )
            champion_capabilities = SandboxCapabilities.model_validate_json(
                canonical_json(champion_sandbox["capabilities"]), strict=True
            )
            challenger_capabilities = SandboxCapabilities.model_validate_json(
                canonical_json(challenger_sandbox["capabilities"]), strict=True
            )
            count = int(document["trace_request_count"])
            if (
                document.get("schema_version") != "sloforge.genesis.evolution.gate-evidence/v1"
                or document.get("stage") != observation.stage.value
                or int(document["seed"]) != observation.deterministic_seed
                or document.get("candidate_id") != candidate_id
                or candidate_id != observation.candidate_id
                or document.get("challenger_capsule_digest") != capsule_digest
                or capsule_digest != observation.capsule_digest
                or document.get("hardware_backed") is not False
                or count != observation.sample_count
                or champion.seed != observation.deterministic_seed
                or candidate.seed != observation.deterministic_seed
                or champion.request_count != count
                or candidate.request_count != count
                or champion_sandbox.get("termination") != "success"
                or challenger_sandbox.get("termination") != "success"
                or champion_sandbox.get("process_group_cleaned") is not True
                or challenger_sandbox.get("process_group_cleaned") is not True
                or any(
                    capability.network_isolation is not IsolationStatus.ENFORCED
                    or capability.filesystem_read_isolation is not IsolationStatus.ENFORCED
                    or capability.filesystem_write_isolation is not IsolationStatus.ENFORCED
                    or capability.environment_sanitization is not IsolationStatus.ENFORCED
                    or capability.child_cleanup is not IsolationStatus.ENFORCED
                    for capability in (champion_capabilities, challenger_capabilities)
                )
            ):
                return False
            champion_cases = {item.request_id: item for item in champion.cases}
            candidate_cases = {item.request_id: item for item in candidate.cases}
            if (
                len(champion_cases) != count
                or len(candidate_cases) != count
                or set(champion_cases) != set(candidate_cases)
                or any(item.token_count != len(item.token_ids) for item in champion.cases)
                or any(item.token_count != len(item.token_ids) for item in candidate.cases)
            ):
                return False
            errors = 0
            interrupted = 0
            mismatches: list[dict[str, object]] = []
            for request_id in sorted(champion_cases):
                expected_case = champion_cases[request_id]
                observed_case = candidate_cases[request_id]
                if expected_case.error is not None or observed_case.error is not None:
                    errors += 1
                    interrupted += 1
                if expected_case.token_ids != observed_case.token_ids:
                    mismatches.append(
                        {
                            "request_id": request_id,
                            "champion_token_ids": list(expected_case.token_ids),
                            "challenger_token_ids": list(observed_case.token_ids),
                        }
                    )
            champion_ttft = _percentile([item.ttft_ns for item in champion.cases], 0.95)
            challenger_ttft = _percentile([item.ttft_ns for item in candidate.cases], 0.95)
            champion_tpot = _percentile([item.mean_tpot_ns for item in champion.cases], 0.99)
            challenger_tpot = _percentile([item.mean_tpot_ns for item in candidate.cases], 0.99)
            if comparison != {
                "mismatches": mismatches,
                "error_count": errors,
                "interrupted_streams": interrupted,
                "champion_p95_ttft_ns": champion_ttft,
                "challenger_p95_ttft_ns": challenger_ttft,
                "champion_p99_mean_tpot_ns": champion_tpot,
                "challenger_p99_mean_tpot_ns": challenger_tpot,
            }:
                return False
            expected = (
                errors / count,
                challenger_ttft / champion_ttft,
                challenger_tpot / champion_tpot,
                len(mismatches) / count,
                interrupted,
            )
            observed = (
                observation.error_rate,
                observation.p95_ttft_ratio,
                observation.p99_tpot_ratio,
                observation.quality_regression,
                observation.interrupted_streams,
            )
            return observed == expected
        except (KeyError, OSError, TypeError, ValueError, ZeroDivisionError):
            return False

    return validate
