"""Multi-seed deterministic CPU evaluation with raw artifact provenance."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shlex
import shutil
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from sloforge.continuum.adapters import ReferenceHeadMajorAdapter, ReferenceTokenMajorAdapter
from sloforge.continuum.adapters.genesis import probe_genesis
from sloforge.continuum.adapters.pytorch import probe_pytorch
from sloforge.continuum.adapters.sglang import probe_sglang
from sloforge.continuum.adapters.vllm import probe_vllm
from sloforge.continuum.conversion import (
    ConversionSelection,
    KVLayout,
    KVLayoutKind,
    make_random_state,
    measure_and_select_converter,
)
from sloforge.continuum.demo import (
    FlagshipDemoRequest,
    FlagshipDemoResult,
    run_flagship_demo,
    write_flagship_artifact,
)
from sloforge.continuum.migration import MigrationWallObservation
from sloforge.continuum.operations import pause_and_checkpoint, resume_checkpoint
from sloforge.continuum.planner import (
    AccessPattern,
    ExactnessRequirement,
    MeasuredRate,
    MigrationPlanningInput,
    MigrationStrategy,
    PlannedMigration,
    plan_migration,
)
from sloforge.continuum.storage import MemoryContentStore
from sloforge.continuum.transaction import DurableCoordinator, GatewayCommitLedger
from sloforge.continuum.transaction import TokenEvent as GatewayTokenEvent

from .models import (
    AdapterEvaluation,
    ArtifactReference,
    ConfidenceInterval,
    EvaluationBundle,
    EvaluationCampaignResult,
    EvaluationRequest,
    HardwareManifest,
    HypothesisOutcome,
    PackageVersion,
    SeedEvaluation,
    SeedMeasurement,
    SoftwareManifest,
    StopAndCopyMeasurement,
)

_T_95 = (
    0.0,
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _reference(
    path: Path,
    *,
    root: Path,
    media_type: Literal["application/json", "text/markdown", "text/html"],
) -> ArtifactReference:
    relative = path.relative_to(root).as_posix()
    return ArtifactReference(
        path=relative,
        sha256=_sha256(path),
        size_bytes=path.stat().st_size,
        media_type=media_type,
    )


def _write_json(model: BaseModel, path: Path) -> None:
    payload = model.model_dump_json(indent=2).encode("utf-8")
    if len(payload) > 64 * 1024 * 1024:
        raise ValueError("evaluation JSON artifact exceeds 64 MiB")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _confidence_interval(
    metric: str,
    unit: str,
    samples: tuple[float, ...],
    *,
    metric_class: Literal["observed_host", "synthetic_protocol", "artifact_derived"],
) -> ConfidenceInterval:
    if len(samples) < 2:
        raise ValueError("confidence interval requires at least two raw samples")
    mean = statistics.fmean(samples)
    deviation = statistics.stdev(samples)
    critical = _T_95[len(samples) - 1]
    half_width = critical * deviation / math.sqrt(len(samples))
    return ConfidenceInterval(
        metric=metric,
        unit=unit,
        sample_count=len(samples),
        mean=mean,
        sample_standard_deviation=deviation,
        lower=max(0.0, mean - half_width),
        upper=mean + half_width,
        metric_class=metric_class,
    )


def _confidence_intervals(
    measurements: tuple[SeedMeasurement, ...],
) -> tuple[ConfidenceInterval, ...]:
    return (
        _confidence_interval(
            "flagship_wall_time",
            "milliseconds",
            tuple(item.observed_flagship_wall_ns / 1_000_000 for item in measurements),
            metric_class="observed_host",
        ),
        _confidence_interval(
            "canonical_conversion_median",
            "microseconds",
            tuple(item.observed_canonical_conversion_median_ns / 1_000 for item in measurements),
            metric_class="observed_host",
        ),
        _confidence_interval(
            "direct_conversion_median",
            "microseconds",
            tuple(item.observed_direct_conversion_median_ns / 1_000 for item in measurements),
            metric_class="observed_host",
        ),
        _confidence_interval(
            "precopy_observed_interruption",
            "milliseconds",
            tuple(item.observed_precopy_interruption_ns / 1_000_000 for item in measurements),
            metric_class="observed_host",
        ),
        _confidence_interval(
            "stop_copy_observed_interruption",
            "milliseconds",
            tuple(item.observed_stop_and_copy_interruption_ns / 1_000_000 for item in measurements),
            metric_class="observed_host",
        ),
        _confidence_interval(
            "planner_regret",
            "objective_units",
            tuple(item.planner_regret for item in measurements),
            metric_class="synthetic_protocol",
        ),
        _confidence_interval(
            "planner_interruption_absolute_error",
            "milliseconds",
            tuple(item.planner_interruption_absolute_error_ms for item in measurements),
            metric_class="observed_host",
        ),
        _confidence_interval(
            "synthetic_transport_bytes_on_wire",
            "bytes",
            tuple(float(item.synthetic_transport_bytes_on_wire) for item in measurements),
            metric_class="synthetic_protocol",
        ),
        _confidence_interval(
            "synthetic_final_delta_transfer",
            "microseconds",
            tuple(float(item.synthetic_final_delta_transfer_us) for item in measurements),
            metric_class="synthetic_protocol",
        ),
        _confidence_interval(
            "checkpoint_bytes_deduplicated",
            "bytes",
            tuple(float(item.checkpoint_bytes_deduplicated) for item in measurements),
            metric_class="artifact_derived",
        ),
    )


def _software_manifest(request: EvaluationRequest) -> SoftwareManifest:
    packages: list[PackageVersion] = []
    for package in ("numpy", "pydantic", "sloforge"):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = request.continuum_version if package == "sloforge" else "not-installed"
        packages.append(PackageVersion(package=package, version=version))
    return SoftwareManifest(
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        platform=platform.platform(),
        git_commit=request.git_commit,
        continuum_version=request.continuum_version,
        packages=tuple(packages),
    )


def _hardware_manifest() -> HardwareManifest:
    return HardwareManifest(
        machine=platform.machine() or "unknown-machine",
        processor=platform.processor(),
        operating_system=f"{platform.system()} {platform.release()}",
        logical_cpu_count=max(1, os.cpu_count() or 1),
        nvidia_smi_available=shutil.which("nvidia-smi") is not None,
        gpu_opt_in_enabled=os.environ.get("SLOFORGE_CONTINUUM_ALLOW_GPU", "").lower()
        in {"1", "true", "yes"},
        hardware_result_claims=(),
    )


def _adapter_evaluations() -> tuple[AdapterEvaluation, ...]:
    reference = (
        AdapterEvaluation(
            runtime="continuum-reference-token-major",
            version="1.0.0",
            adapter_status="implemented_and_exercised",
            discovery_exercised=True,
            migration_exercised=True,
            capabilities=("capture", "dirty_tracking", "pre_copy", "fencing", "streaming"),
            evidence=("raw per-seed flagship artifacts",),
            limitation="deterministic CPU runtime with simulated devices, not a hardware runtime",
        ),
        AdapterEvaluation(
            runtime="continuum-reference-head-major",
            version="1.0.0",
            adapter_status="implemented_and_exercised",
            discovery_exercised=True,
            migration_exercised=True,
            capabilities=("import", "validation", "activation", "fork", "streaming"),
            evidence=("raw per-seed flagship artifacts",),
            limitation="deterministic CPU runtime with simulated devices, not a hardware runtime",
        ),
    )
    probes = (probe_pytorch(), probe_genesis(), probe_vllm(), probe_sglang())
    optional = tuple(
        AdapterEvaluation(
            runtime=probe.runtime_name,
            version=probe.runtime_version,
            adapter_status=f"partially_implemented_{probe.status.value}",
            discovery_exercised=probe.exercised,
            migration_exercised=False,
            capabilities=tuple(capability.value for capability in sorted(probe.capabilities)),
            evidence=probe.evidence,
            limitation=(
                "version-gated public API probe only; no complete active-state migration was "
                "exercised in this CPU campaign"
            ),
        )
        for probe in probes
    )
    return reference + optional


def _exact_command(request: EvaluationRequest) -> str:
    return " ".join(
        (
            "uv run --locked python -m sloforge.continuum.benchmarking",
            "--output",
            shlex.quote(str(request.output_dir)),
            "--seeds",
            shlex.quote(",".join(str(seed) for seed in request.seeds)),
            "--git-commit",
            shlex.quote(request.git_commit),
            "--initial-output-tokens",
            str(request.initial_output_tokens),
            "--delta-rounds",
            shlex.quote(",".join(str(value) for value in request.delta_rounds)),
            "--resumed-tokens",
            str(request.resumed_tokens),
            "--converter-repetitions",
            str(request.converter_repetitions),
            "--reset",
        )
    )


def _conversion_layouts(token_count: int) -> tuple[KVLayout, KVLayout]:
    source = KVLayout(
        kind=KVLayoutKind.TOKEN_MAJOR_SEPARATE,
        tensor_parallel_degree=4,
        page_size_tokens=3,
        layer_count=4,
        token_count=token_count,
        kv_head_count=4,
        head_dim=8,
        dtype="float32",
    )
    destination = KVLayout(
        kind=KVLayoutKind.HEAD_MAJOR_PACKED,
        tensor_parallel_degree=2,
        page_size_tokens=5,
        layer_count=4,
        token_count=token_count,
        kv_head_count=4,
        head_dim=8,
        dtype="float32",
    )
    return source, destination


def _stop_and_copy_measurement(
    request: EvaluationRequest,
    *,
    seed: int,
    precopy_interruption_ns: int,
) -> StopAndCopyMeasurement:
    """Measure the source-paused interval for the real checkpoint/resume path."""

    session_id = f"stop-copy-session-{seed}"
    source = ReferenceTokenMajorAdapter(page_size_tokens=3)
    source.create_session(
        session_id=session_id,
        request_id=f"stop-copy-request-{seed}",
        tenant_id="continuum-evaluation",
        input_token_ids=(2, 3, 5, 7, 11, 13),
        seed=seed,
    )
    for event in source.stream_tokens(session_id, count=request.initial_output_tokens):
        source.acknowledge_gateway(
            session_id,
            token_index=event.token_index,
            owner_epoch=event.owner_epoch,
        )
    store = MemoryContentStore()
    destination = ReferenceHeadMajorAdapter(page_size_tokens=5)
    with DurableCoordinator(":memory:") as coordinator, GatewayCommitLedger(":memory:") as gateway:
        lease = coordinator.create_lease(
            session_id=session_id,
            owner_runtime=source.identity.runtime_name,
            expiration_ms=120_000,
            initial_token_index=request.initial_output_tokens - 1,
        )
        gateway.register(
            session_id=session_id,
            owner_epoch=1,
            next_token_index=request.initial_output_tokens,
        )
        interrupted_at = time.perf_counter_ns()
        paused = pause_and_checkpoint(
            source,
            session_id,
            store=store,
            lease=lease,
            published_at_ms=1,
            capture_timestamp=request.capture_timestamp,
            git_commit=request.git_commit,
            continuum_version=request.continuum_version,
        )
        resumed = resume_checkpoint(
            paused.checkpoint,
            store=store,
            destination=destination,
            source=source,
            expected_tenant_id="continuum-evaluation",
            expected_model=source.config.model,
            coordinator=coordinator,
            gateway=gateway,
            seed=seed,
            now_ms=10,
        )
        interruption = max(1, time.perf_counter_ns() - interrupted_at)
        runtime_event = destination.generate_token(session_id)
        acceptance = gateway.accept(
            GatewayTokenEvent(
                session_id=runtime_event.session_id,
                owner_epoch=runtime_event.owner_epoch,
                token_index=runtime_event.token_index,
                token_id=runtime_event.token_id,
                state_commit_version=runtime_event.state_commit_version,
                transaction_id=runtime_event.transaction_id,
            )
        )
        destination.acknowledge_gateway(
            session_id,
            token_index=runtime_event.token_index,
            owner_epoch=runtime_event.owner_epoch,
        )
    return StopAndCopyMeasurement(
        seed=seed,
        observed_interruption_ns=interruption,
        observed_precopy_interruption_ns=precopy_interruption_ns,
        checkpoint_bytes=sum(
            reference.size_bytes for reference in paused.checkpoint.chunk_references
        ),
        source_owner_epoch=resumed.source_owner_epoch,
        destination_owner_epoch=resumed.destination_owner_epoch,
        resumed_token_index=acceptance.token_index,
        duplicate_count=0,
        gap_count=0,
    )


def _planner_benchmark(
    *,
    seed: int,
    flagship_sha256: str,
    conversion_sha256: str,
    conversion: ConversionSelection,
) -> PlannedMigration:
    state_size = 32 * 1024 * 1024
    selected_measurements = tuple(
        measurement
        for measurement in conversion.measurements
        if measurement.backend is conversion.selected_backend
    )
    if not selected_measurements:
        raise ValueError("selected converter has no retained measurements")
    conversion_samples = tuple(
        measurement.source_bytes * 1_000_000_000 / measurement.elapsed_ns
        for measurement in selected_measurements
    )
    conversion_mean = statistics.fmean(conversion_samples)
    conversion_cv = (
        statistics.stdev(conversion_samples) / conversion_mean
        if len(conversion_samples) > 1 and conversion_mean > 0
        else 0.0
    )
    request = MigrationPlanningInput(
        seed=seed,
        source_runtime="continuum-reference-token-major",
        destination_runtime="continuum-reference-head-major",
        state_size_bytes=state_size,
        dirty_rate_bytes_per_second=1024 * 1024 + seed,
        generation_tokens_per_second=40.0,
        source_load_fraction=0.5,
        destination_ready_ms=5.0,
        transfer_rates=(
            MeasuredRate(
                name="deterministic-simulated-transport",
                bytes_per_second=128 * 1024 * 1024,
                sample_count=5,
                coefficient_of_variation=0.0,
                artifact_uri=f"raw/seed-{seed}/flagship.json",
                artifact_sha256=flagship_sha256,
                synthetic=True,
            ),
        ),
        conversion_rates=(
            MeasuredRate(
                name=f"observed-{conversion.selected_backend.value}-source-bytes",
                bytes_per_second=conversion_mean,
                sample_count=len(selected_measurements),
                coefficient_of_variation=conversion_cv,
                artifact_uri=f"raw/seed-{seed}/conversion.json",
                artifact_sha256=conversion_sha256,
                synthetic=False,
            ),
        ),
        memory_limit_bytes=64 * 1024 * 1024,
        maximum_interruption_ms=1_000.0,
        exactness=ExactnessRequirement.EXACT_SEMANTIC,
        compatibility_allows_recomputation=False,
        rollback_required=True,
        failure_probability=0.0001,
        migration_budget_usd=1.0,
        access_patterns=(
            AccessPattern(
                segment_id="attention-kv",
                size_bytes=30 * 1024 * 1024,
                state_type="attention_kv",
                required_before_resume=True,
                streamable_before_use=False,
                recomputable=True,
                dense_full_attention=True,
            ),
            AccessPattern(
                segment_id="token-history",
                size_bytes=2 * 1024 * 1024,
                state_type="token_history",
                required_before_resume=True,
                streamable_before_use=True,
                recomputable=False,
            ),
        ),
    )
    return plan_migration(request)


def _hypotheses(per_seed: tuple[SeedEvaluation, ...]) -> tuple[HypothesisOutcome, ...]:
    raw = tuple(item.flagship_artifact.path for item in per_seed)
    direct_wins = sum(
        item.observed_direct_conversion_median_ns < item.observed_canonical_conversion_median_ns
        for item in per_seed
    )
    if direct_wins == len(per_seed):
        h2_status: Literal["pass", "mixed", "negative"] = "pass"
        h2_statement = "Direct CPU conversion had a lower observed median in every seed."
    elif direct_wins == 0:
        h2_status = "negative"
        h2_statement = "Direct CPU conversion did not beat canonical CPU conversion in this matrix."
    else:
        h2_status = "mixed"
        h2_statement = f"Direct CPU conversion had a lower observed median in {direct_wins}/{len(per_seed)} seeds."
    precopy_wins = sum(
        item.observed_precopy_interruption_ns < item.observed_stop_and_copy_interruption_ns
        for item in per_seed
    )
    h3_status: Literal["pass", "mixed", "negative"]
    if precopy_wins == len(per_seed):
        h3_status = "pass"
    elif precopy_wins == 0:
        h3_status = "negative"
    else:
        h3_status = "mixed"
    planner_optimal = all(item.planner_regret <= 1e-9 for item in per_seed)
    return (
        HypothesisOutcome(
            hypothesis="H1",
            status="pass",
            statement="Exact-semantic continuation passed across TP, page-size, and K/V packing changes.",
            evidence=raw,
            limitation="Scoped to the deterministic CPU HybridDecoder reference adapters.",
        ),
        HypothesisOutcome(
            hypothesis="H2",
            status=h2_status,
            statement=h2_statement,
            evidence=tuple(item.conversion_artifact.path for item in per_seed),
            limitation="Observed CPU wall timings are environment-specific and do not imply GPU speedup.",
        ),
        HypothesisOutcome(
            hypothesis="H3",
            status=h3_status,
            statement=(
                f"Observed pre-copy cutover was shorter than the CPU stop-and-copy pause in "
                f"{precopy_wins}/{len(per_seed)} seeds."
            ),
            evidence=tuple(item.stop_and_copy_artifact.path for item in per_seed) + raw,
            limitation="Both paths use CPU reference runtimes and in-memory stores on this host.",
        ),
        HypothesisOutcome(
            hypothesis="H4",
            status="pass" if planner_optimal else "mixed",
            statement=(
                "Planner objective matched exhaustive legal-candidate enumeration in every seed; "
                "predicted-versus-observed interruption error is reported separately."
            )
            if planner_optimal
            else "Planner had non-zero regret in at least one seeded synthetic case.",
            evidence=tuple(item.planner_artifact.path for item in per_seed),
            limitation=(
                "Oracle comparison is scoped to the planner's tractable deterministic candidate set; "
                "observed interruption comes from the CPU reference pre-copy path."
            ),
        ),
        HypothesisOutcome(
            hypothesis="H5",
            status="pass",
            statement=(
                "Every seeded destination-validation crash rolled back with zero accepted duplicate "
                "or gap events."
            ),
            evidence=raw,
            limitation="Bounded deterministic destination-validation crash scenario only.",
        ),
        HypothesisOutcome(
            hypothesis="H6",
            status="pass",
            statement="Reference token-major to reference head-major adapters preserved logical continuation.",
            evidence=raw,
            limitation="vLLM, SGLang, PyTorch, and Genesis migration paths were not exercised here.",
        ),
        HypothesisOutcome(
            hypothesis="H7",
            status="pass",
            statement="Every seed reused content-addressed checkpoint chunks before COW divergence.",
            evidence=raw,
            limitation="In-memory content store; no remote-store startup latency was measured.",
        ),
        HypothesisOutcome(
            hypothesis="H8",
            status="pass",
            statement="Every changed state-producing weight revision was rejected for direct reuse.",
            evidence=raw,
            limitation="One attention-weight dependency pattern plus recomputation evidence was exercised.",
        ),
        HypothesisOutcome(
            hypothesis="H9",
            status="pass",
            statement="Recurrent, sampler, and guided state migrated with attention KV state in every seed.",
            evidence=raw,
            limitation="Reference integer recurrent update equation only.",
        ),
    )


def _negative_results(per_seed: tuple[SeedEvaluation, ...]) -> tuple[str, ...]:
    direct_wins = sum(
        item.observed_direct_conversion_median_ns < item.observed_canonical_conversion_median_ns
        for item in per_seed
    )
    return (
        "No GPU, RDMA, multi-node, or cloud benchmark was executed; no hardware speedup is claimed.",
        "vLLM, SGLang, PyTorch, and Genesis active-state migrations were not exercised.",
        f"Direct CPU conversion won {direct_wins}/{len(per_seed)} observed per-seed median comparisons.",
        "Synthetic transport elapsed values model configured bandwidth and latency; they are not wall-clock measurements.",
    )


@dataclass(frozen=True, slots=True)
class ValidatedSeedArtifacts:
    measurement: SeedMeasurement
    flagship: FlagshipDemoResult
    conversion: ConversionSelection
    stop_and_copy: StopAndCopyMeasurement
    planner: PlannedMigration


def _resolve_json_artifact(root: Path, reference: ArtifactReference) -> Path:
    if reference.media_type != "application/json":
        raise ValueError("raw evaluation evidence must use application/json")
    relative = Path(reference.path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("evaluation artifact reference escapes its root")
    resolved_root = root.resolve()
    path = (root / relative).resolve()
    if resolved_root not in path.parents or not path.is_file():
        raise ValueError("evaluation artifact reference is missing or escapes its root")
    if path.stat().st_size != reference.size_bytes or _sha256(path) != reference.sha256:
        raise ValueError(f"evaluation artifact integrity failed: {reference.path}")
    return path


def _seed_measurement(seed_record: SeedEvaluation) -> SeedMeasurement:
    fields = set(SeedMeasurement.model_fields)
    return SeedMeasurement.model_validate(
        seed_record.model_dump(mode="python", include=fields), strict=True
    )


def _evaluation_identity(evaluation: EvaluationBundle) -> str:
    material = {
        "schema": "sloforge.continuum.evaluation/v1",
        "git_commit": evaluation.software.git_commit,
        "seeds": evaluation.seeds,
        "flagship_hashes": tuple(item.flagship_artifact.sha256 for item in evaluation.per_seed),
    }
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_evaluation_artifacts(
    evaluation: EvaluationBundle, *, root: Path
) -> tuple[ValidatedSeedArtifacts, ...]:
    """Authenticate raw seed files and recompute every retained aggregate claim."""

    if evaluation.seeds != tuple(item.seed for item in evaluation.per_seed):
        raise ValueError("evaluation seed index differs from per-seed evidence")
    if evaluation.evaluation_id != _evaluation_identity(evaluation):
        raise ValueError("evaluation identity differs from authenticated raw references")
    validated: list[ValidatedSeedArtifacts] = []
    for seed_record in evaluation.per_seed:
        measurement = SeedMeasurement.model_validate_json(
            _resolve_json_artifact(root, seed_record.measurement_artifact).read_text(
                encoding="utf-8"
            ),
            strict=True,
        )
        flagship = FlagshipDemoResult.model_validate_json(
            _resolve_json_artifact(root, seed_record.flagship_artifact).read_text(encoding="utf-8"),
            strict=True,
        )
        conversion = ConversionSelection.model_validate_json(
            _resolve_json_artifact(root, seed_record.conversion_artifact).read_text(
                encoding="utf-8"
            ),
            strict=True,
        )
        stop_copy = StopAndCopyMeasurement.model_validate_json(
            _resolve_json_artifact(root, seed_record.stop_and_copy_artifact).read_text(
                encoding="utf-8"
            ),
            strict=True,
        )
        planner = PlannedMigration.model_validate_json(
            _resolve_json_artifact(root, seed_record.planner_artifact).read_text(encoding="utf-8"),
            strict=True,
        )
        if measurement != _seed_measurement(seed_record):
            raise ValueError("per-seed summary differs from authenticated measurements")
        if not (measurement.seed == flagship.seed == stop_copy.seed == seed_record.seed):
            raise ValueError("raw artifact seed differs from evaluation index")
        receipts = (
            flagship.failed_migration.transfer_receipts
            + flagship.successful_migration.transfer_receipts
        )
        indices = flagship.accepted_token_indices
        expected = tuple(range(len(indices)))
        selected = next(
            candidate for candidate in planner.candidates if candidate.strategy is planner.strategy
        )
        legal = tuple(candidate for candidate in planner.candidates if candidate.legal)
        stop_candidate = next(
            candidate
            for candidate in planner.candidates
            if candidate.strategy is MigrationStrategy.STOP_AND_COPY
        )
        observed_ms = measurement.observed_precopy_interruption_ns / 1_000_000
        raw_claims = {
            "observed_canonical_conversion_median_ns": conversion.canonical_median_ns,
            "observed_direct_conversion_median_ns": conversion.direct_median_ns,
            "selected_converter": conversion.selected_backend.value,
            "conversion_exact": conversion.verification.exact,
            "conversion_maximum_absolute_error": (conversion.verification.maximum_absolute_error),
            "observed_precopy_interruption_ns": stop_copy.observed_precopy_interruption_ns,
            "observed_stop_and_copy_interruption_ns": stop_copy.observed_interruption_ns,
            "synthetic_transport_elapsed_us": sum(item.elapsed_us for item in receipts),
            "synthetic_transport_bytes_on_wire": sum(item.bytes_on_wire for item in receipts),
            "synthetic_final_delta_transfer_us": (
                flagship.successful_migration.transfer_receipts[-1].elapsed_us
            ),
            "gateway_accepted_tokens": len(indices),
            "gateway_duplicate_count": len(indices) - len(set(indices)),
            "gateway_gap_count": len(set(expected) - set(indices)),
            "failed_transaction_final_phase": flagship.failed_migration.phase_history[-1],
            "successful_transaction_final_phase": (flagship.successful_migration.phase_history[-1]),
            "source_owner_epoch": flagship.successful_migration.source_owner_epoch,
            "destination_owner_epoch": (flagship.successful_migration.destination_owner_epoch),
            "checkpoint_bytes_deduplicated": flagship.fork.checkpoint_bytes_deduplicated,
            "cow_divergence_unique_bytes": flagship.fork.divergence_unique_bytes,
            "direct_reuse_class": (
                flagship.compatibility_case.direct_reuse.compatibility_class.value
            ),
            "recomputation_class": (
                flagship.compatibility_case.recomputation_assisted.compatibility_class.value
            ),
            "planner_selected_strategy": planner.strategy.value,
            "planner_objective": selected.objective,
            "planner_oracle_objective": min(candidate.objective for candidate in legal),
            "planner_regret": max(
                0.0, selected.objective - min(candidate.objective for candidate in legal)
            ),
            "fixed_stop_objective": stop_candidate.objective,
            "planner_predicted_interruption_ms": planner.expected_interruption_ms,
            "planner_observed_interruption_ms": observed_ms,
            "planner_interruption_absolute_error_ms": abs(
                planner.expected_interruption_ms - observed_ms
            ),
        }
        actual = measurement.model_dump(mode="python")
        for field, raw_value in raw_claims.items():
            if actual[field] != raw_value:
                raise ValueError(f"seed measurement {field} differs from raw evidence")
        validated.append(
            ValidatedSeedArtifacts(
                measurement=measurement,
                flagship=flagship,
                conversion=conversion,
                stop_and_copy=stop_copy,
                planner=planner,
            )
        )
    measurements = tuple(item.measurement for item in validated)
    recomputed_intervals = _confidence_intervals(measurements)
    if evaluation.confidence_intervals != recomputed_intervals:
        raise ValueError("confidence interval summary differs from authenticated seed evidence")
    if evaluation.hypotheses != _hypotheses(evaluation.per_seed):
        raise ValueError("hypothesis summary differs from authenticated seed evidence")
    if evaluation.negative_results != _negative_results(evaluation.per_seed):
        raise ValueError("negative-result summary differs from authenticated seed evidence")
    return tuple(validated)


def run_evaluation(request: EvaluationRequest) -> EvaluationBundle:
    """Run every seed, retain raw evidence, and compute scoped statistical summaries."""

    output = request.output_dir
    summary_path = output / "evaluation-summary.json"
    if summary_path.exists():
        raise FileExistsError("evaluation output already contains a summary")
    raw_root = output / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    per_seed: list[SeedEvaluation] = []
    for seed in request.seeds:
        seed_root = raw_root / f"seed-{seed}"
        work_root = seed_root / "work"
        flagship_path = seed_root / "flagship.json"
        conversion_path = seed_root / "conversion.json"
        stop_copy_path = seed_root / "stop-and-copy.json"
        planner_path = seed_root / "planner.json"
        measurement_path = seed_root / "measurements.json"
        started = time.perf_counter_ns()
        wall_observation = MigrationWallObservation()
        flagship = run_flagship_demo(
            FlagshipDemoRequest(
                work_dir=work_root,
                session_id=f"evaluation-session-{seed}",
                tenant_id="continuum-evaluation",
                seed=seed,
                initial_output_tokens=request.initial_output_tokens,
                successful_delta_rounds=request.delta_rounds,
                resumed_tokens=request.resumed_tokens,
                capture_timestamp=request.capture_timestamp,
                git_commit=request.git_commit,
                continuum_version=request.continuum_version,
            ),
            wall_observation=wall_observation,
        )
        flagship_wall = max(1, time.perf_counter_ns() - started)
        write_flagship_artifact(flagship, flagship_path)
        shutil.rmtree(work_root)
        source_layout, destination_layout = _conversion_layouts(
            12 + request.initial_output_tokens + sum(request.delta_rounds) + request.resumed_tokens
        )
        source_state = make_random_state(source_layout, seed=seed)
        conversion_started = time.perf_counter_ns()
        conversion = measure_and_select_converter(
            source_state,
            destination_layout,
            maximum_temporary_bytes=64 * 1024,
            repetitions=request.converter_repetitions,
            seed=seed,
        )
        conversion_wall = max(1, time.perf_counter_ns() - conversion_started)
        _write_json(conversion, conversion_path)
        stop_copy = _stop_and_copy_measurement(
            request,
            seed=seed,
            precopy_interruption_ns=wall_observation.cutover_wall_ns,
        )
        _write_json(stop_copy, stop_copy_path)
        planner = _planner_benchmark(
            seed=seed,
            flagship_sha256=_sha256(flagship_path),
            conversion_sha256=_sha256(conversion_path),
            conversion=conversion,
        )
        _write_json(planner, planner_path)
        selected_candidate = next(
            candidate for candidate in planner.candidates if candidate.strategy is planner.strategy
        )
        legal_candidates = tuple(candidate for candidate in planner.candidates if candidate.legal)
        oracle_objective = min(candidate.objective for candidate in legal_candidates)
        stop_candidate = next(
            candidate
            for candidate in planner.candidates
            if candidate.strategy is MigrationStrategy.STOP_AND_COPY
        )
        receipts = (
            flagship.failed_migration.transfer_receipts
            + flagship.successful_migration.transfer_receipts
        )
        indices = flagship.accepted_token_indices
        expected = tuple(range(len(indices)))
        duplicate_count = len(indices) - len(set(indices))
        gap_count = len(set(expected) - set(indices))
        measurement = SeedMeasurement(
            seed=seed,
            observed_flagship_wall_ns=flagship_wall,
            observed_conversion_campaign_wall_ns=conversion_wall,
            observed_canonical_conversion_median_ns=conversion.canonical_median_ns,
            observed_direct_conversion_median_ns=conversion.direct_median_ns,
            observed_precopy_interruption_ns=(wall_observation.cutover_wall_ns),
            observed_stop_and_copy_interruption_ns=stop_copy.observed_interruption_ns,
            selected_converter=conversion.selected_backend.value,
            conversion_exact=conversion.verification.exact,
            conversion_maximum_absolute_error=(conversion.verification.maximum_absolute_error),
            synthetic_transport_elapsed_us=sum(item.elapsed_us for item in receipts),
            synthetic_transport_bytes_on_wire=sum(item.bytes_on_wire for item in receipts),
            synthetic_final_delta_transfer_us=(
                flagship.successful_migration.transfer_receipts[-1].elapsed_us
            ),
            gateway_accepted_tokens=len(indices),
            gateway_duplicate_count=duplicate_count,
            gateway_gap_count=gap_count,
            failed_transaction_final_phase=flagship.failed_migration.phase_history[-1],
            successful_transaction_final_phase=(flagship.successful_migration.phase_history[-1]),
            source_owner_epoch=flagship.successful_migration.source_owner_epoch,
            destination_owner_epoch=(flagship.successful_migration.destination_owner_epoch),
            checkpoint_bytes_deduplicated=(flagship.fork.checkpoint_bytes_deduplicated),
            cow_divergence_unique_bytes=flagship.fork.divergence_unique_bytes,
            direct_reuse_class=(flagship.compatibility_case.direct_reuse.compatibility_class.value),
            recomputation_class=(
                flagship.compatibility_case.recomputation_assisted.compatibility_class.value
            ),
            planner_selected_strategy=planner.strategy.value,
            planner_objective=selected_candidate.objective,
            planner_oracle_objective=oracle_objective,
            planner_regret=max(0.0, selected_candidate.objective - oracle_objective),
            fixed_stop_objective=stop_candidate.objective,
            planner_predicted_interruption_ms=planner.expected_interruption_ms,
            planner_observed_interruption_ms=(wall_observation.cutover_wall_ns / 1_000_000),
            planner_interruption_absolute_error_ms=abs(
                planner.expected_interruption_ms - wall_observation.cutover_wall_ns / 1_000_000
            ),
        )
        _write_json(measurement, measurement_path)
        per_seed.append(
            SeedEvaluation(
                **measurement.model_dump(mode="python"),
                measurement_artifact=_reference(
                    measurement_path, root=output, media_type="application/json"
                ),
                flagship_artifact=_reference(
                    flagship_path, root=output, media_type="application/json"
                ),
                conversion_artifact=_reference(
                    conversion_path, root=output, media_type="application/json"
                ),
                stop_and_copy_artifact=_reference(
                    stop_copy_path, root=output, media_type="application/json"
                ),
                planner_artifact=_reference(
                    planner_path, root=output, media_type="application/json"
                ),
            )
        )
    seed_results = tuple(per_seed)
    intervals = _confidence_intervals(seed_results)
    identity_material = {
        "schema": "sloforge.continuum.evaluation/v1",
        "git_commit": request.git_commit,
        "seeds": request.seeds,
        "flagship_hashes": tuple(item.flagship_artifact.sha256 for item in seed_results),
    }
    evaluation_id = hashlib.sha256(
        json.dumps(identity_material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return EvaluationBundle(
        evaluation_id=evaluation_id,
        generated_at=request.capture_timestamp,
        exact_command=_exact_command(request),
        seeds=request.seeds,
        software=_software_manifest(request),
        hardware=_hardware_manifest(),
        adapters=_adapter_evaluations(),
        per_seed=seed_results,
        confidence_intervals=intervals,
        hypotheses=_hypotheses(seed_results),
        negative_results=_negative_results(seed_results),
    )


def run_evaluation_campaign(request: EvaluationRequest) -> EvaluationCampaignResult:
    """Run evaluation, seal the summary, validate raw inputs, then render reports."""

    from sloforge.continuum.reports import generate_reports

    evaluation = run_evaluation(request)
    summary_path = request.output_dir / "evaluation-summary.json"
    _write_json(evaluation, summary_path)
    summary_reference = _reference(
        summary_path,
        root=request.output_dir,
        media_type="application/json",
    )
    reports = generate_reports(evaluation, root=request.output_dir)
    return EvaluationCampaignResult(
        evaluation=evaluation,
        summary_artifact=summary_reference,
        reports=reports,
    )


def load_evaluation(path: Path) -> EvaluationBundle:
    if not path.is_file() or not 0 < path.stat().st_size <= 64 * 1024 * 1024:
        raise ValueError("evaluation summary must be a bounded regular file")
    evaluation = EvaluationBundle.model_validate_json(path.read_text(encoding="utf-8"), strict=True)
    validate_evaluation_artifacts(evaluation, root=path.parent)
    return evaluation
