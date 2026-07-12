"""End-to-end CPU serving evidence for a generated HybridDecoder kernel."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Literal, cast

from sloforge.genesis.frontend import inspect_reference_package, load_reference_package
from sloforge.genesis.frontend.models import InspectionResult
from sloforge.genesis.runtime import generate_baseline_runtime
from sloforge.genesis.runtime.generator import generated_runtime_template_hashes
from sloforge.genesis.sandbox import (
    SandboxLimits,
    SandboxRequest,
    SandboxTermination,
    execute_sandboxed,
)
from sloforge.genesis.verification import (
    BenchmarkContract,
    EvidenceStatus,
    MetricDirection,
    evaluate_performance,
)
from sloforge.util import sha256_file

from .benchmark import validate_benchmark_report
from .executor import validate_correctness_evidence
from .generator import generate_candidates, validate_generated_source
from .models import (
    AcceptanceStatus,
    CandidateDecision,
    CorrectnessEvidence,
    KernelBenchmarkReport,
    KernelCandidate,
    LabStatus,
    RuntimeBundleIdentity,
    RuntimeImpactConfig,
    RuntimeImpactReport,
    RuntimeImpactSample,
    RuntimeImpactStatistics,
    RuntimeImpactValidation,
    RuntimeRequestSemantics,
)

_REFERENCE_FUNCTION = '''def quantized_state_update(previous: int, activation: float) -> int:
    """Declared custom state transform: symmetric int8 round-to-nearest."""

    combined = previous * 0.625 + activation * 31.0
    return max(-127, min(127, round(combined)))
'''
_PATCHED_FUNCTION = '''def quantized_state_update(previous: int, activation: float) -> int:
    """Generated scalar adapter over the synthesized vector state-update kernel."""

    return quantized_recurrent_state_update(
        [previous], [activation], 1, 0, 1, 0, 1, False
    )[0]
'''
_PATCH_IMPORT = "from generated_kernel import quantized_recurrent_state_update\n"


def derive_runtime_impact_config(seed: int, **updates: object) -> RuntimeImpactConfig:
    """Derive disjoint deterministic seed domains from one public campaign seed."""

    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 1 << 64:
        raise ValueError("runtime-impact campaign seed must be an unsigned 64-bit integer")

    def derive(label: str) -> int:
        return int.from_bytes(
            hashlib.sha256(f"kernel-runtime-impact\0{seed}\0{label}".encode()).digest()[:8],
            "big",
        )

    values: dict[str, object] = {
        "synthesis_seed": seed,
        "runtime_generation_seed": derive("runtime-generation"),
        "trace_seed": derive("trace"),
        "trial_order_seed": derive("trial-order"),
        "bootstrap_seed": derive("bootstrap"),
        # PYTHONHASHSEED, which the trusted sandbox binds to this value, has a
        # narrower unsigned 32-bit domain than the other campaign seeds.
        "sandbox_seed": derive("sandbox") & 0xFFFFFFFF,
    }
    values.update(updates)
    return RuntimeImpactConfig.model_validate(values, strict=True)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def _copy_reference_package(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"runtime-impact package destination already exists: {destination}")
    if source.is_symlink():
        raise ValueError("runtime-impact source package must not be a symlink")
    for entry in source.rglob("*"):
        if entry.is_symlink():
            raise ValueError(f"runtime-impact source package contains a symlink: {entry}")
    shutil.copytree(source, destination, symlinks=False)


def _materialize_patched_package(
    source_package: Path,
    destination: Path,
    candidate: KernelCandidate,
    source: str,
) -> None:
    diagnostics = validate_generated_source(candidate, source)
    if diagnostics:
        raise ValueError(f"generated kernel did not pass its source allowlist: {diagnostics}")
    _copy_reference_package(source_package, destination)
    reference_path = destination / "reference.py"
    reference = reference_path.read_text(encoding="utf-8")
    if reference.count(_REFERENCE_FUNCTION) != 1:
        raise ValueError("flagship reference state-update function no longer matches its contract")
    if reference.count("import math\n") != 1:
        raise ValueError("flagship reference import anchor is ambiguous")
    reference = reference.replace("import math\n", f"import math\n\n{_PATCH_IMPORT}", 1)
    reference = reference.replace(_REFERENCE_FUNCTION, _PATCHED_FUNCTION, 1)
    reference_path.write_text(reference, encoding="utf-8")
    (destination / "generated_kernel.py").write_text(source, encoding="utf-8")
    manifest_path = destination / "reference_package.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("flagship reference manifest must be an object")
    auxiliary = manifest.get("auxiliary_modules", [])
    if not isinstance(auxiliary, list) or auxiliary:
        raise ValueError("flagship package unexpectedly declares auxiliary modules")
    manifest["auxiliary_modules"] = ["generated_kernel.py"]
    semantic_contract = manifest.get("semantic_contract")
    if not isinstance(semantic_contract, dict):
        raise TypeError("flagship package semantic contract must be an object")
    allowed_control_flow = semantic_contract.get("allowed_control_flow")
    if allowed_control_flow != ["if", "for"]:
        raise ValueError("flagship control-flow contract changed unexpectedly")
    semantic_contract["allowed_control_flow"] = ["if", "for", "while"]
    manifest_path.write_bytes(_canonical(manifest) + b"\n")


def _trace_payload(package_root: Path, config: RuntimeImpactConfig) -> bytes:
    package = load_reference_package(package_root)
    corpus_path = package.resolve(package.manifest.quality_contract.search_corpus)
    corpus: list[dict[str, object]] = []
    for line in corpus_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if not isinstance(item, dict):
                raise TypeError("search corpus rows must be objects")
            corpus.append(item)
    if not corpus:
        raise ValueError("flagship search corpus is empty")
    generator = random.Random(config.trace_seed)
    requests: list[dict[str, object]] = []
    for index in range(config.request_count):
        sample = corpus[index % len(corpus)]
        suffix = chr(ord("a") + generator.randrange(26)) * (index % 3)
        maximum_value = sample["maximum_new_tokens"]
        if not isinstance(maximum_value, int) or isinstance(maximum_value, bool):
            raise TypeError("search corpus maximum_new_tokens must be an integer")
        maximum = maximum_value
        requests.append(
            {
                "request_id": f"kernel-impact-{index:02d}",
                "text": f"{sample['text']}{suffix}",
                "maximum_new_tokens": max(1, min(8, maximum + (index % 2))),
                "seed": int.from_bytes(
                    hashlib.sha256(
                        f"{config.trace_seed}\0request\0{index}".encode()
                    ).digest()[:8],
                    "big",
                ),
                "batching_eligible": index % 4 != 3,
            }
        )
    return _canonical(
        {
            "schema_version": "sloforge.genesis.kernel-runtime-trace/v1",
            "trace_seed": config.trace_seed,
            "interleaved_submission": True,
            "requests": requests,
        }
    ) + b"\n"


def _software_manifest() -> tuple[str, ...]:
    affinity = (
        ",".join(str(cpu) for cpu in sorted(os.sched_getaffinity(0)))
        if hasattr(os, "sched_getaffinity")
        else "unavailable"
    )
    return (
        f"python={platform.python_version()}",
        f"implementation={platform.python_implementation()}",
        f"machine={platform.machine() or 'unknown'}",
        f"system={platform.system()}",
        f"release={platform.release()}",
        f"logical_cpu_count={os.cpu_count() or 0}",
        f"cpu_affinity={affinity}",
        "timer=perf_counter_ns",
        "gpu_execution=false",
    )


def _hardware_fingerprint(manifest: tuple[str, ...]) -> str:
    hardware = tuple(
        item
        for item in manifest
        if item.startswith(
            ("machine=", "system=", "release=", "logical_cpu_count=", "cpu_affinity=")
        )
    )
    if len(hardware) != 5:
        raise ValueError("runtime-impact hardware identity is incomplete")
    return "local-cpu:" + hashlib.sha256("|".join(hardware).encode()).hexdigest()


def _bundle_identity(
    alternative: Literal["reference", "candidate"],
    bundle_root: Path,
    package_root: Path,
    inspection_path: Path,
) -> RuntimeBundleIdentity:
    package = load_reference_package(package_root)
    artifact_manifest_path = bundle_root / "artifact_manifest.json"
    artifact_manifest = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(artifact_manifest, dict):
        raise TypeError("generated runtime artifact manifest must be an object")
    return RuntimeBundleIdentity(
        alternative=alternative,
        bundle_root=str(bundle_root.resolve()),
        runtime_id=str(artifact_manifest["runtime_id"]),
        package_root=str(package_root.resolve()),
        package_hash=package.package_hash,
        inspection_path=str(inspection_path.resolve()),
        inspection_sha256=sha256_file(inspection_path),
        artifact_manifest_path=str(artifact_manifest_path.resolve()),
        artifact_manifest_sha256=sha256_file(artifact_manifest_path),
    )


def _sandbox_execute(
    *,
    runner: Path,
    execution_config: Path,
    execution_inputs: Path,
    output_root: Path,
    config: RuntimeImpactConfig,
    mode: str,
) -> tuple[Path, SandboxTermination, str]:
    result_path = output_root / "runner-output.json"
    repository_python = Path(__file__).resolve().parents[3]
    result = execute_sandboxed(
        SandboxRequest(
            argv=(
                sys.executable,
                str(runner.resolve(strict=True)),
                "--config",
                str(execution_config.resolve(strict=True)),
                "--output",
                str(result_path.resolve()),
                "--mode",
                mode,
            ),
            working_directory=execution_inputs.resolve(strict=True),
            read_only_paths=(
                execution_inputs.resolve(strict=True),
                repository_python.resolve(strict=True),
                Path(sys.prefix),
                Path(sys.base_prefix),
            ),
            artifact_output_directory=output_root.resolve(),
            seed=config.sandbox_seed,
            limits=SandboxLimits(
                wall_time_seconds=config.sandbox_wall_time_seconds,
                cpu_time_seconds=max(1, int(config.sandbox_wall_time_seconds)),
                memory_bytes=2 * 1024 * 1024 * 1024,
                process_count=1,
                output_bytes=64 * 1024,
                artifact_bytes=16 * 1024 * 1024,
                artifact_entries=64,
                open_files=64,
            ),
        )
    )
    if not result.succeeded or not result_path.is_file():
        raise RuntimeError(
            f"sandboxed runtime-impact {mode} failed: "
            f"{result.termination.value}: {result.stderr}"
        )
    if not result.process_group_cleaned:
        raise RuntimeError("runtime-impact sandbox did not clean its process group")
    return result_path, result.termination, result.capabilities.backend.value


def _parse_runner_output(path: Path, *, mode: str) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "mode",
        "samples",
        "runtime_observations",
        "semantics",
    }:
        raise ValueError("runtime-impact runner output has an invalid shape")
    if value["schema_version"] != "sloforge.genesis.kernel-runtime-runner/v1":
        raise ValueError("runtime-impact runner schema is unsupported")
    if value["mode"] != mode:
        raise ValueError("runtime-impact runner mode does not match its invocation")
    return value


def _typed_semantics(
    runner: dict[str, object], alternative: str
) -> tuple[RuntimeRequestSemantics, ...]:
    semantics = runner["semantics"]
    if not isinstance(semantics, dict) or not isinstance(semantics.get(alternative), list):
        raise ValueError("runtime-impact semantics are incomplete")
    return tuple(
        RuntimeRequestSemantics.model_validate_json(
            json.dumps(item, sort_keys=True, separators=(",", ":")), strict=True
        )
        for item in semantics[alternative]
    )


def _runtime_observations(runner: dict[str, object]) -> tuple[dict[str, object], ...]:
    observations = runner["runtime_observations"]
    if not isinstance(observations, list) or not all(
        isinstance(item, dict) for item in observations
    ):
        raise ValueError("runtime observations must be a bounded object list")
    return tuple(cast("dict[str, object]", item) for item in observations)


def _semantic_matches(
    runner: dict[str, object],
    reference: tuple[RuntimeRequestSemantics, ...],
    candidate: tuple[RuntimeRequestSemantics, ...],
) -> tuple[bool, bool]:
    expected_tokens = {
        "reference": tuple(
            {"request_id": item.request_id, "token_ids": list(item.token_ids)}
            for item in reference
        ),
        "candidate": tuple(
            {"request_id": item.request_id, "token_ids": list(item.token_ids)}
            for item in candidate
        ),
    }
    output_exact = True
    observations = _runtime_observations(runner)
    by_trial: dict[int, dict[str, object]] = {}
    for observation in observations:
        alternative = observation.get("alternative")
        trial_index = observation.get("trial_index")
        requests = observation.get("requests")
        if alternative not in {"reference", "candidate"} or not isinstance(trial_index, int):
            raise ValueError("runtime observation identity is invalid")
        if not isinstance(requests, list):
            raise ValueError("runtime observation requests must be a list")
        if tuple(requests) != expected_tokens[alternative]:
            output_exact = False
        by_trial.setdefault(trial_index, {})[alternative] = requests
    for pair in by_trial.values():
        if pair.get("reference") != pair.get("candidate"):
            output_exact = False
    state_exact = reference == candidate
    return output_exact, state_exact


def _statistics(
    samples: tuple[RuntimeImpactSample, ...],
    config: RuntimeImpactConfig,
    *,
    workload_fingerprint: str,
    hardware_fingerprint: str,
    software_manifest: tuple[str, ...],
    semantic_valid: bool,
) -> RuntimeImpactStatistics:
    ordered = tuple(sorted(samples, key=lambda item: item.order_index))
    if tuple(item.order_index for item in ordered) != tuple(range(config.repetitions * 2)):
        raise ValueError("runtime-impact order indices are not complete")
    for alternative in ("reference", "candidate"):
        trials = sorted(item.trial_index for item in ordered if item.alternative == alternative)
        if trials != list(range(config.repetitions)):
            raise ValueError(f"runtime-impact {alternative} trials are incomplete")
    reference = tuple(float(item.duration_ns) for item in ordered if item.alternative == "reference")
    candidate = tuple(float(item.duration_ns) for item in ordered if item.alternative == "candidate")
    contract = BenchmarkContract(
        benchmark_id="hybrid-qstate-generated-runtime-end-to-end",
        metric="trace_completion_ns",
        unit="ns",
        direction=MetricDirection.LOWER_IS_BETTER,
        workload_fingerprint=workload_fingerprint,
        hardware_fingerprint=hardware_fingerprint,
        software_manifest_hash=hashlib.sha256("|".join(software_manifest).encode()).hexdigest(),
        warmup_count=config.warmup_count,
        practical_significance_percent=config.practical_significance_percent,
        noise_floor_percent=config.noise_floor_percent,
        bootstrap_rounds=config.bootstrap_rounds,
        confidence=config.confidence,
    )
    evidence = evaluate_performance(
        contract,
        reference,
        candidate,
        seed=config.bootstrap_seed,
        run_order=tuple(
            "baseline" if item.alternative == "reference" else "candidate" for item in ordered
        ),
    )
    mapped = {
        EvidenceStatus.PASSED: LabStatus.PASSED,
        EvidenceStatus.FAILED: LabStatus.FAILED,
        EvidenceStatus.INCONCLUSIVE: LabStatus.INCONCLUSIVE,
        EvidenceStatus.UNAVAILABLE: LabStatus.UNAVAILABLE,
    }[evidence.status]
    status = mapped if semantic_valid else LabStatus.FAILED
    rationale = (
        evidence.rationale
        if semantic_valid
        else "exact generated-runtime output or persistent-state semantics did not match"
    )
    return RuntimeImpactStatistics(
        status=status,
        reference_median_ns=evidence.baseline_median,
        candidate_median_ns=evidence.candidate_median,
        improvement_percent=evidence.improvement_percent,
        confidence_interval_low_percent=evidence.interval_low_percent,
        confidence_interval_high_percent=evidence.interval_high_percent,
        effect_size=evidence.effect_size,
        practical_significance_percent=config.practical_significance_percent,
        noise_floor_percent=config.noise_floor_percent,
        rationale=rationale,
    )


def _raw_sample_bytes(samples: tuple[RuntimeImpactSample, ...]) -> bytes:
    return b"".join(_canonical(item.model_dump(mode="json")) + b"\n" for item in samples)


def _artifact_path(
    artifact_root: Path,
    value: str | Path,
    *,
    kind: Literal["file", "directory"],
) -> Path:
    """Resolve one recorded artifact without trusting a report-supplied path."""

    if artifact_root.is_symlink():
        raise ValueError("runtime-impact artifact root must not be a symlink")
    root = artifact_root.resolve(strict=True)
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        relative_spelling = candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("runtime-impact artifact path escapes its trusted root") from error
    if any(part in {"", ".", ".."} for part in relative_spelling.parts):
        raise ValueError("runtime-impact artifact path is not normalized")
    cursor = root
    for part in relative_spelling.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError("runtime-impact artifact path contains a symlink")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("runtime-impact artifact path escapes its trusted root") from error
    if kind == "file" and not resolved.is_file():
        raise ValueError("runtime-impact artifact is not a regular file")
    if kind == "directory" and not resolved.is_dir():
        raise ValueError("runtime-impact artifact is not a directory")
    return resolved


def _verify_runtime_bundle(
    identity: RuntimeBundleIdentity,
    *,
    artifact_root: Path,
    runtime_generation_seed: int,
) -> None:
    package_root = _artifact_path(artifact_root, identity.package_root, kind="directory")
    bundle = _artifact_path(artifact_root, identity.bundle_root, kind="directory")
    inspection_path = _artifact_path(artifact_root, identity.inspection_path, kind="file")
    manifest_path = _artifact_path(
        artifact_root, identity.artifact_manifest_path, kind="file"
    )
    package = load_reference_package(package_root)
    if package.package_hash != identity.package_hash:
        raise ValueError("runtime-impact package hash changed")
    if sha256_file(inspection_path) != identity.inspection_sha256:
        raise ValueError("runtime-impact inspection artifact changed")
    if sha256_file(manifest_path) != identity.artifact_manifest_sha256:
        raise ValueError("runtime-impact generated runtime manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("runtime_id") != identity.runtime_id:
        raise ValueError("runtime-impact runtime identity changed")
    if manifest.get("package_hash") != identity.package_hash:
        raise ValueError("runtime-impact runtime/package binding changed")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("runtime-impact runtime artifact list is missing")
    for item in artifacts:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("runtime-impact runtime artifact record is invalid")
        relative, digest = item
        path = (bundle / str(relative)).resolve(strict=True)
        path.relative_to(bundle)
        if sha256_file(path) != digest:
            raise ValueError("runtime-impact generated runtime artifact changed")
    expected_files = {
        "artifact_manifest.json",
        "correctness_harness.py",
        "deployment_manifest.json",
        "runtime.py",
        "runtime_config.json",
    }
    actual_files: set[str] = set()
    for entry in bundle.rglob("*"):
        if entry.is_symlink():
            raise ValueError("runtime-impact bundle contains a symlink")
        if entry.is_dir():
            continue
        if not entry.is_file():
            raise ValueError("runtime-impact bundle contains a non-regular artifact")
        actual_files.add(entry.relative_to(bundle).as_posix())
    if actual_files != expected_files:
        raise ValueError("runtime-impact generated runtime closure is not exact")
    for name, digest in generated_runtime_template_hashes().items():
        if sha256_file(bundle / name) != digest:
            raise ValueError("runtime-impact bundle does not use a trusted runtime template")
    runtime_config = json.loads((bundle / "runtime_config.json").read_text(encoding="utf-8"))
    if not isinstance(runtime_config, dict):
        raise TypeError("runtime-impact runtime configuration must be an object")
    if runtime_config.get("generation_seed") != runtime_generation_seed:
        raise ValueError("runtime-impact runtime generation seed changed")
    genome_hash = runtime_config.get("genome_hash")
    if genome_hash is not None and not isinstance(genome_hash, str):
        raise TypeError("runtime-impact runtime genome hash must be a string or null")
    inspection = InspectionResult.model_validate_json(inspection_path.read_bytes(), strict=True)
    with tempfile.TemporaryDirectory(prefix="sloforge-kernel-runtime-validate-") as temporary:
        regenerated = Path(temporary) / "runtime"
        generate_baseline_runtime(
            package_root,
            inspection,
            regenerated,
            seed=runtime_generation_seed,
            genome_hash=genome_hash,
        )
        for name in expected_files:
            if (bundle / name).read_bytes() != (regenerated / name).read_bytes():
                raise ValueError("runtime-impact runtime differs from trusted regeneration")


def _validate_recorded_replay(report: RuntimeImpactReport, *, artifact_root: Path) -> None:
    validation = report.validation
    if validation is None:
        return
    replay_path = _artifact_path(
        artifact_root, validation.replay_output_path, kind="file"
    )
    if sha256_file(replay_path) != validation.replay_output_sha256:
        raise ValueError("runtime-impact independent replay artifact changed")
    replay = _parse_runner_output(replay_path, mode="replay")
    reference = _typed_semantics(replay, "reference")
    candidate = _typed_semantics(replay, "candidate")
    output_exact, state_exact = _semantic_matches(replay, reference, candidate)
    if not output_exact or not state_exact:
        raise ValueError("runtime-impact independent replay did not preserve exact semantics")
    if reference != report.reference_semantics or candidate != report.candidate_semantics:
        raise ValueError("runtime-impact independent replay differs from measured semantics")


def validate_runtime_impact_report(
    report: RuntimeImpactReport, *, artifact_root: Path
) -> None:
    """Reopen raw artifacts and independently reconstruct every report claim."""

    candidate_source_path = _artifact_path(
        artifact_root, report.candidate_source_path, kind="file"
    )
    trace_path = _artifact_path(artifact_root, report.trace_path, kind="file")
    runner_path = _artifact_path(artifact_root, report.runner_path, kind="file")
    runner_output_path = _artifact_path(
        artifact_root, report.runner_output_path, kind="file"
    )
    raw_samples_path = _artifact_path(
        artifact_root, report.raw_samples_path, kind="file"
    )
    if sha256_file(candidate_source_path) != report.candidate_source_sha256:
        raise ValueError("runtime-impact generated candidate source changed")
    if report.candidate.source_sha256 != report.candidate_source_sha256:
        raise ValueError("runtime-impact candidate identity/source binding changed")
    generated = {
        item.candidate_id: (item, source)
        for item, source in generate_candidates(seed=report.config.synthesis_seed)
    }
    expected_candidate = generated.get(report.candidate.candidate_id)
    if expected_candidate is None or expected_candidate[0] != report.candidate:
        raise ValueError("runtime-impact candidate is not from the deterministic generator")
    if expected_candidate[1].encode() != candidate_source_path.read_bytes():
        raise ValueError("runtime-impact candidate source is not generator-derived")
    if sha256_file(trace_path) != report.trace_sha256:
        raise ValueError("runtime-impact trace changed")
    if hashlib.sha256(trace_path.read_bytes()).hexdigest() != report.workload_fingerprint:
        raise ValueError("runtime-impact workload fingerprint is not trace-derived")
    if _hardware_fingerprint(report.software_manifest) != report.hardware_fingerprint:
        raise ValueError("runtime-impact hardware fingerprint is not manifest-derived")
    for identity in report.runtime_bundles:
        _verify_runtime_bundle(
            identity,
            artifact_root=artifact_root,
            runtime_generation_seed=report.config.runtime_generation_seed,
        )
    identities = {item.alternative: item for item in report.runtime_bundles}
    source_root = _artifact_path(
        artifact_root, identities["reference"].package_root, kind="directory"
    )
    candidate_root = _artifact_path(
        artifact_root, identities["candidate"].package_root, kind="directory"
    )
    source = load_reference_package(source_root)
    candidate = load_reference_package(candidate_root)
    canonical_flagship = load_reference_package(
        Path(__file__).resolve().parents[4] / "models/reference_tasks/hybrid_decoder"
    )
    if source.package_hash != canonical_flagship.package_hash:
        raise ValueError("runtime-impact source package is not the trusted flagship package")
    if source.package_hash != report.source_package_hash:
        raise ValueError("runtime-impact source package identity changed")
    if candidate.package_hash != report.patched_package_hash:
        raise ValueError("runtime-impact patched package identity changed")
    with tempfile.TemporaryDirectory(prefix="sloforge-kernel-package-validate-") as temporary:
        reconstructed = Path(temporary) / "candidate-package"
        _materialize_patched_package(
            source_root,
            reconstructed,
            report.candidate,
            candidate_source_path.read_text(encoding="utf-8"),
        )
        if load_reference_package(reconstructed).package_hash != candidate.package_hash:
            raise ValueError("runtime-impact patched package is not deterministically reconstructed")
    trusted_runner_sha256 = sha256_file(Path(__file__).with_name("runtime_impact_runner.py"))
    if report.runner_sha256 != trusted_runner_sha256:
        raise ValueError("runtime-impact runner is not the installed trusted runner")
    if sha256_file(runner_path) != report.runner_sha256:
        raise ValueError("runtime-impact trusted runner changed")
    if sha256_file(runner_output_path) != report.runner_output_sha256:
        raise ValueError("runtime-impact measured runner output changed")
    raw = raw_samples_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != report.raw_samples_sha256:
        raise ValueError("runtime-impact raw samples changed")
    if raw != _raw_sample_bytes(report.samples):
        raise ValueError("runtime-impact raw sample file is not report-derived")
    measured = _parse_runner_output(runner_output_path, mode="measure")
    parsed_samples = measured["samples"]
    if not isinstance(parsed_samples, list):
        raise ValueError("runtime-impact measured samples are not a list")
    rebuilt_samples = tuple(
        RuntimeImpactSample.model_validate(item, strict=True) for item in parsed_samples
    )
    if rebuilt_samples != report.samples:
        raise ValueError("runtime-impact embedded samples differ from sandbox output")
    observation_outputs: dict[tuple[str, int], object] = {}
    for observation in _runtime_observations(measured):
        alternative = observation.get("alternative")
        trial_index = observation.get("trial_index")
        requests = observation.get("requests")
        if alternative not in {"reference", "candidate"} or not isinstance(trial_index, int):
            raise ValueError("runtime-impact observation identity is invalid")
        key = (alternative, trial_index)
        if key in observation_outputs:
            raise ValueError("runtime-impact observation identity is duplicated")
        observation_outputs[key] = requests
    for sample in report.samples:
        requests = observation_outputs.get((sample.alternative, sample.trial_index))
        if requests is None:
            raise ValueError("runtime-impact sample has no recorded runtime observation")
        if hashlib.sha256(_canonical(requests)).hexdigest() != sample.output_sha256:
            raise ValueError("runtime-impact sample output digest is not observation-derived")
    reference_semantics = _typed_semantics(measured, "reference")
    candidate_semantics = _typed_semantics(measured, "candidate")
    if reference_semantics != report.reference_semantics:
        raise ValueError("runtime-impact reference semantics changed")
    if candidate_semantics != report.candidate_semantics:
        raise ValueError("runtime-impact candidate semantics changed")
    output_exact, state_exact = _semantic_matches(
        measured, reference_semantics, candidate_semantics
    )
    if (output_exact, state_exact) != (report.output_exact_match, report.state_exact_match):
        raise ValueError("runtime-impact semantic claims are not replay-derived")
    statistics = _statistics(
        report.samples,
        report.config,
        workload_fingerprint=report.workload_fingerprint,
        hardware_fingerprint=report.hardware_fingerprint,
        software_manifest=report.software_manifest,
        semantic_valid=output_exact and state_exact,
    )
    if statistics != report.statistics:
        raise ValueError("runtime-impact statistics are not raw-sample-derived")
    _validate_recorded_replay(report, artifact_root=artifact_root)


def benchmark_generated_runtime_impact(
    candidate: KernelCandidate,
    source: str,
    correctness: CorrectnessEvidence,
    *,
    reference_package_root: Path,
    output_root: Path,
    config: RuntimeImpactConfig,
) -> RuntimeImpactReport:
    """Patch, inspect, generate, sandbox, measure, independently replay, and persist."""

    validate_correctness_evidence(correctness)
    if correctness.status is not LabStatus.PASSED:
        raise ValueError("runtime impact requires a correctness-passed kernel candidate")
    if candidate != correctness.candidate or candidate.deterministic_seed != config.synthesis_seed:
        raise ValueError("runtime-impact candidate does not match its correctness/config evidence")
    if output_root.exists() and (output_root.is_symlink() or any(output_root.iterdir())):
        raise ValueError("runtime-impact output must be a new empty directory")
    output_root.mkdir(parents=True, exist_ok=True)
    execution_inputs = output_root / "execution-inputs"
    execution_inputs.mkdir()
    reference_package = execution_inputs / "reference-package"
    patched_package = execution_inputs / "candidate-package"
    _copy_reference_package(reference_package_root, reference_package)
    _materialize_patched_package(reference_package_root, patched_package, candidate, source)
    source_identity = load_reference_package(reference_package)
    patched_identity = load_reference_package(patched_package)
    if source_identity.package_hash == patched_identity.package_hash:
        raise RuntimeError("generated kernel patch did not change the package identity")
    reference_inspection_path = execution_inputs / "reference-inspection.json"
    candidate_inspection_path = execution_inputs / "candidate-inspection.json"
    reference_inspection = inspect_reference_package(
        reference_package, use_torch_export=False, output_path=reference_inspection_path
    )
    candidate_inspection = inspect_reference_package(
        patched_package, use_torch_export=False, output_path=candidate_inspection_path
    )
    reference_bundle = execution_inputs / "reference-runtime"
    candidate_bundle = execution_inputs / "candidate-runtime"
    generate_baseline_runtime(
        reference_package,
        reference_inspection,
        reference_bundle,
        seed=config.runtime_generation_seed,
    )
    generate_baseline_runtime(
        patched_package,
        candidate_inspection,
        candidate_bundle,
        seed=config.runtime_generation_seed,
    )
    trace_path = execution_inputs / "interleaved-trace.json"
    _write_new(trace_path, _trace_payload(reference_package, config))
    runner = execution_inputs / "runtime-impact-runner.py"
    shutil.copyfile(Path(__file__).with_name("runtime_impact_runner.py"), runner)
    execution_config = execution_inputs / "execution-config.json"
    _write_new(
        execution_config,
        _canonical(
            {
                **config.model_dump(mode="json"),
                "reference_bundle": str(reference_bundle.resolve()),
                "candidate_bundle": str(candidate_bundle.resolve()),
                "reference_package": str(reference_package.resolve()),
                "candidate_package": str(patched_package.resolve()),
                "trace_path": str(trace_path.resolve()),
            }
        )
        + b"\n",
    )
    sandbox_output = output_root / "measurement-sandbox"
    measured_path, termination, backend = _sandbox_execute(
        runner=runner,
        execution_config=execution_config,
        execution_inputs=execution_inputs,
        output_root=sandbox_output,
        config=config,
        mode="measure",
    )
    measured = _parse_runner_output(measured_path, mode="measure")
    raw_samples = measured["samples"]
    if not isinstance(raw_samples, list):
        raise ValueError("runtime-impact measurement returned invalid raw samples")
    samples = tuple(RuntimeImpactSample.model_validate(item, strict=True) for item in raw_samples)
    reference_semantics = _typed_semantics(measured, "reference")
    candidate_semantics = _typed_semantics(measured, "candidate")
    output_exact, state_exact = _semantic_matches(
        measured, reference_semantics, candidate_semantics
    )
    software_manifest = _software_manifest()
    hardware_fingerprint = _hardware_fingerprint(software_manifest)
    workload_fingerprint = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    statistics = _statistics(
        samples,
        config,
        workload_fingerprint=workload_fingerprint,
        hardware_fingerprint=hardware_fingerprint,
        software_manifest=software_manifest,
        semantic_valid=output_exact and state_exact,
    )
    raw_samples_path = output_root / "raw-runtime-samples.jsonl"
    _write_new(raw_samples_path, _raw_sample_bytes(samples))
    identities = (
        _bundle_identity(
            "reference", reference_bundle, reference_package, reference_inspection_path
        ),
        _bundle_identity(
            "candidate", candidate_bundle, patched_package, candidate_inspection_path
        ),
    )
    provisional = RuntimeImpactReport(
        candidate=candidate,
        config=config,
        source_package_hash=source_identity.package_hash,
        patched_package_hash=patched_identity.package_hash,
        candidate_source_path=str((patched_package / "generated_kernel.py").resolve()),
        candidate_source_sha256=sha256_file(patched_package / "generated_kernel.py"),
        trace_path=str(trace_path.resolve()),
        trace_sha256=sha256_file(trace_path),
        workload_fingerprint=workload_fingerprint,
        hardware_fingerprint=hardware_fingerprint,
        software_manifest=software_manifest,
        runtime_bundles=identities,
        samples=samples,
        raw_samples_path=str(raw_samples_path.resolve()),
        raw_samples_sha256=sha256_file(raw_samples_path),
        runner_output_path=str(measured_path.resolve()),
        runner_output_sha256=sha256_file(measured_path),
        runner_path=str(runner.resolve()),
        runner_sha256=sha256_file(runner),
        reference_semantics=reference_semantics,
        candidate_semantics=candidate_semantics,
        output_exact_match=output_exact,
        state_exact_match=state_exact,
        statistics=statistics,
        sandbox_termination=termination.value,
        sandbox_backend=backend,
    )
    validate_runtime_impact_report(provisional, artifact_root=output_root)
    replay_root = output_root / "independent-validation-sandbox"
    replay_path, _replay_termination, replay_backend = _sandbox_execute(
        runner=runner,
        execution_config=execution_config,
        execution_inputs=execution_inputs,
        output_root=replay_root,
        config=config,
        mode="replay",
    )
    validation = RuntimeImpactValidation(
        replay_output_path=str(replay_path.resolve()),
        replay_output_sha256=sha256_file(replay_path),
        sandbox_termination="success",
        sandbox_backend=replay_backend,
    )
    report = provisional.model_copy(update={"validation": validation})
    validate_runtime_impact_report(report, artifact_root=output_root)
    report_path = output_root / "runtime-impact-report.json"
    _write_new(report_path, report.model_dump_json(indent=2).encode() + b"\n")
    return report


def decide_with_runtime_impact(
    correctness: CorrectnessEvidence,
    benchmark: KernelBenchmarkReport,
    impact: RuntimeImpactReport,
    *,
    artifact_root: Path,
) -> CandidateDecision:
    """Apply the full serving gate; isolated timing can never substitute for it."""

    validate_correctness_evidence(correctness)
    validate_benchmark_report(benchmark)
    validate_runtime_impact_report(impact, artifact_root=artifact_root)
    if not (correctness.candidate == benchmark.candidate == impact.candidate):
        raise ValueError("kernel decision evidence refers to different candidates")
    by_name = {item.regime: item for item in benchmark.regimes}
    micro = by_name.get("micro_noncontiguous")
    operator_loop = by_name.get("operator_loop_batch_32")
    micro_status = micro.status if micro is not None else LabStatus.UNAVAILABLE
    operator_status = operator_loop.status if operator_loop is not None else LabStatus.UNAVAILABLE
    all_pass = (
        correctness.status is LabStatus.PASSED
        and micro_status is LabStatus.PASSED
        and operator_status is LabStatus.PASSED
        and all(item.status is LabStatus.PASSED for item in benchmark.regimes)
        and impact.statistics.status is LabStatus.PASSED
        and impact.output_exact_match
        and impact.state_exact_match
        and impact.validation is not None
    )
    if all_pass:
        status = AcceptanceStatus.ACCEPTED
        claim = "CPU generated-runtime speedup accepted within the declared local evidence scope"
        reasons = (
            "exact token and persistent-state semantics passed independent sandbox replay",
            "the end-to-end confidence interval exceeded noise and practical thresholds",
            "this is local CPU evidence and is not a GPU or production-hardware claim",
        )
    else:
        status = (
            AcceptanceStatus.REJECTED
            if impact.statistics.status is LabStatus.FAILED
            or correctness.status is LabStatus.FAILED
            or benchmark.status is LabStatus.FAILED
            else AcceptanceStatus.INCONCLUSIVE
        )
        claim = "no speedup claim; end-to-end generated-runtime acceptance gate did not pass"
        reasons = (
            "exact semantics passed" if impact.output_exact_match and impact.state_exact_match else "exact semantics failed",
            f"end-to-end statistical gate was {impact.statistics.status.value}",
            "isolated operator results cannot override the generated-runtime serving gate",
        )
    return CandidateDecision(
        candidate_id=correctness.candidate.candidate_id,
        status=status,
        correctness_status=correctness.status,
        microbenchmark_status=micro_status,
        operator_loop_status=operator_status,
        full_stack_status=impact.statistics.status,
        claim=claim,
        reasons=reasons,
    )


__all__ = [
    "benchmark_generated_runtime_impact",
    "decide_with_runtime_impact",
    "derive_runtime_impact_config",
    "validate_runtime_impact_report",
]
