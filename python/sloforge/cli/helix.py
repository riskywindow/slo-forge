from __future__ import annotations

import hashlib
import importlib
import json
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Literal, TypeVar, cast

import typer
from pydantic import BaseModel
from rich.console import Console

from sloforge.helix.capture import CoordinatedBranchPoint
from sloforge.helix.credit import (
    BranchOutcome,
    BranchRelativeCredit,
    assign_branch_relative_credit,
)
from sloforge.helix.datasets import (
    ReferenceTrainingBatchManifest,
    build_reference_training_batch,
)
from sloforge.helix.demo import run_cpu_demo
from sloforge.helix.evaluation import (
    EvaluationRun,
    run_reference_evaluation,
    write_evaluation_reports,
)
from sloforge.helix.faults import (
    FaultKind,
    FaultObservation,
    FaultRunner,
    InjectedFault,
    canonical_digest,
    compile_fault_plan,
    load_fault_plan_request,
)
from sloforge.helix.policy import DeterministicPolicy
from sloforge.helix.promotion import (
    GateEvidence,
    PolicyRegistry,
    TrustedPolicyPromotionCapsule,
    validate_policy_promotion_capsule,
)
from sloforge.helix.replay import ReplayMode, ReplayTrace, compare_replay
from sloforge.helix.rewards import RewardRun
from sloforge.helix.rollouts import ReferenceTrajectory
from sloforge.helix.scheduler import (
    SchedulerPolicy,
    SchedulerRequest,
    compile_resource_plan,
    load_scheduler_request,
)
from sloforge.helix.trainers import ReferenceTrainer, TrainingAlgorithm
from sloforge.helix.transactions import LearningTransactionStore

helix_app = typer.Typer(
    help="Capture, branch, train, validate, and transactionally promote Helix policies.",
    no_args_is_help=True,
)
policy_app = typer.Typer(help="Inspect versioned policy routing.")
branchpoint_app = typer.Typer(help="Validate coordinated model/environment branch points.")
trajectory_app = typer.Typer(help="Validate trajectory provenance and event chains.")
dataset_app = typer.Typer(help="Validate provenance-complete training batches.")
credit_app = typer.Typer(help="Assign structured counterfactual sibling credit.")
transaction_app = typer.Typer(help="Inspect durable learning transactions.")
scheduler_app = typer.Typer(help="Compile learning-aware resource schedules.")
lineage_app = typer.Typer(help="Query portable Helix lineage graphs.")
fault_app = typer.Typer(help="Run bounded deterministic Helix fault campaigns.")
branch_trace_app = typer.Typer(help="Capture canonical branch and state-operation traces.")
characterize_app = typer.Typer(
    help="Run restartable, evidence-labelled BranchFabric characterization studies."
)
helix_app.add_typer(policy_app, name="policy")
helix_app.add_typer(branchpoint_app, name="branchpoint")
helix_app.add_typer(trajectory_app, name="trajectory")
helix_app.add_typer(dataset_app, name="dataset")
helix_app.add_typer(credit_app, name="credit")
helix_app.add_typer(transaction_app, name="transaction")
helix_app.add_typer(scheduler_app, name="scheduler")
helix_app.add_typer(lineage_app, name="lineage")
helix_app.add_typer(fault_app, name="fault")
helix_app.add_typer(branch_trace_app, name="trace")
helix_app.add_typer(characterize_app, name="characterize")
console = Console()
ModelT = TypeVar("ModelT", bound=BaseModel)
CharacterizationHardware = Literal["cpu", "gpu"]

_SESSION_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")
_WORKFLOW_MODULE = "sloforge.helix.characterization.workflow"
_MAX_ANALYSIS_EVENTS = 10_000_000


def _json(value: object) -> None:
    console.print_json(json.dumps(value, default=str))


def _workflow_callback(name: str) -> Callable[..., object]:
    """Load the integrating workflow only when a command actually needs it."""

    try:
        module = importlib.import_module(_WORKFLOW_MODULE)
    except ModuleNotFoundError as error:
        if error.name != _WORKFLOW_MODULE:
            raise
        raise typer.BadParameter(
            "the characterization workflow backend is not installed in this checkout; "
            "run `make branchfabric-trace-check` to validate available trace studies, "
            "or install/integrate sloforge.helix.characterization.workflow"
        ) from error
    callback = getattr(module, name, None)
    if callback is None or not callable(callback):
        raise typer.BadParameter(
            f"the characterization workflow backend does not provide {name}(); "
            "update the backend and CLI together"
        )
    return cast(Callable[..., object], callback)


def _emit_backend_result(result: object) -> None:
    if isinstance(result, BaseModel):
        _json(result.model_dump(mode="json"))
        return
    _json(result)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _analysis_target(output: Path, default_name: str) -> Path:
    return output if output.suffix else output / default_name


def _check_replaceable_json(target: Path, *, schema_version: str, replace: bool) -> None:
    if not target.exists():
        return
    if not target.is_file() or target.is_symlink():
        raise typer.BadParameter(f"refusing to replace non-regular output {target}")
    if not replace:
        raise typer.BadParameter(f"output {target} exists; pass --replace to replace it")
    try:
        prior = json.loads(target.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise typer.BadParameter(f"refusing to replace unrecognized output {target}") from error
    if not isinstance(prior, dict) or prior.get("schema_version") != schema_version:
        raise typer.BadParameter(
            f"refusing to replace {target}: expected prior {schema_version} artifact"
        )


def _write_analysis(
    output: Path,
    *,
    default_name: str,
    schema_version: str,
    payload: dict[str, object],
    replace: bool,
) -> Path:
    target = _analysis_target(output, default_name)
    _check_replaceable_json(target, schema_version=schema_version, replace=replace)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return target


def _parse_positive_ints(
    raw: str,
    *,
    option: str,
    maximum: int,
    require_first_one: bool = False,
) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as error:
        raise typer.BadParameter(f"{option} must be comma-separated integers") from error
    if not values or any(value < 1 or value > maximum for value in values):
        raise typer.BadParameter(f"{option} values must be within 1..{maximum}")
    if len(values) != len(set(values)) or tuple(sorted(values)) != values:
        raise typer.BadParameter(f"{option} values must be unique and strictly increasing")
    if require_first_one and values[0] != 1:
        raise typer.BadParameter(f"{option} must begin with 1")
    return values


def _parse_page_sizes(raw: str) -> tuple[int, ...]:
    multipliers = {"k": 1024, "m": 1024 * 1024}
    values: list[int] = []
    for item in raw.split(","):
        normalized = item.strip().lower()
        match = re.fullmatch(r"([1-9][0-9]*)([km]?)", normalized)
        if match is None:
            raise typer.BadParameter(
                "--page-sizes must contain values such as 4k,16k,64k,256k,1m,2m"
            )
        value = int(match.group(1)) * multipliers.get(match.group(2), 1)
        if not 4096 <= value <= 1024 * 1024 * 1024:
            raise typer.BadParameter("--page-sizes values must be within 4 KiB..1 GiB")
        values.append(value)
    parsed = tuple(values)
    if not parsed or len(parsed) != len(set(parsed)):
        raise typer.BadParameter("--page-sizes must be non-empty and unique")
    return parsed


def _load_gate(path: Path, expected: str) -> GateEvidence:
    gate = GateEvidence.model_validate_json(path.read_bytes(), strict=True)
    if gate.gate != expected:
        raise typer.BadParameter(f"expected {expected!r} evidence, found {gate.gate!r}")
    return gate


@branch_trace_app.command("branch")
def trace_branch_command(
    session: Annotated[str, typer.Option("--session")],
    output: Annotated[Path, typer.Option("--output", "-o")],
    seed: Annotated[int, typer.Option("--seed", min=0, max=2**63 - 1)] = 41,
    trace_level: Annotated[
        Literal["disabled", "minimal", "full"], typer.Option("--trace-level")
    ] = "full",
    buffer_capacity_events: Annotated[
        int, typer.Option("--buffer-capacity-events", min=1, max=_MAX_ANALYSIS_EVENTS)
    ] = 100_000,
    hardware_baseline: Annotated[
        Path, typer.Option("--hardware-baseline", exists=True, dir_okay=False)
    ] = Path("artifacts/branchfabric/manifests/hardware-baseline.json"),
    software_baseline: Annotated[
        Path, typer.Option("--software-baseline", exists=True, dir_okay=False)
    ] = Path("artifacts/branchfabric/manifests/software-baseline.json"),
    replace: Annotated[bool, typer.Option("--replace")] = False,
) -> None:
    """Trace the bounded reference Helix session into both canonical v1 streams."""

    if _SESSION_PATTERN.fullmatch(session) is None:
        raise typer.BadParameter(
            "--session must be a 1-128 character alphanumeric, dot, dash, or underscore ID"
        )
    if session != "production-session":
        raise typer.BadParameter(
            "this backend currently traces only the bounded reference session "
            "`production-session`; arbitrary live-session attachment is not implemented"
        )
    from sloforge.helix.characterization.runner import run_vertical_trace
    from sloforge.helix.characterization.trace import TraceLevel

    target = output / f"{session}-seed-{seed}"
    try:
        result = run_vertical_trace(
            target,
            seed=seed,
            trace_level=TraceLevel(trace_level),
            buffer_capacity_events=buffer_capacity_events,
            replace=replace,
            hardware_baseline=hardware_baseline,
            software_baseline=software_baseline,
        )
    except (FileExistsError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    _json(
        {
            "session": session,
            "output": target.as_posix(),
            **result.model_dump(mode="json"),
        }
    )


@characterize_app.command("run")
def characterize_run_command(
    matrix: Annotated[Path, typer.Option("--matrix", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "artifacts/branchfabric/characterization/cpu-reference"
    ),
    hardware: Annotated[CharacterizationHardware, typer.Option("--hardware")] = "cpu",
    seed: Annotated[int, typer.Option("--seed", min=0, max=2**63 - 1)] = 20260809,
    max_experiments: Annotated[
        int, typer.Option("--max-experiments", min=1, max=100_000)
    ] = 100_000,
    timeout_seconds: Annotated[
        float, typer.Option("--timeout-seconds", min=1.0, max=86_400.0)
    ] = 300.0,
    replace: Annotated[bool, typer.Option("--replace")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Execute a bounded matrix, or validate and expand it with --dry-run."""

    from sloforge.helix.characterization.matrix import expand_matrix, load_matrix

    try:
        loaded = load_matrix(matrix)
        cases = expand_matrix(loaded)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    if len(cases) > max_experiments:
        raise typer.BadParameter(
            f"matrix expands to {len(cases)} cases, above --max-experiments={max_experiments}"
        )
    if dry_run:
        _json(
            {
                "dry_run": True,
                "matrix_id": loaded.matrix_id,
                "case_count": len(cases),
                "hardware": hardware,
                "seed": seed,
                "timeout_seconds": timeout_seconds,
                "distribution_claim": loaded.distribution_claim,
                "output": output.as_posix(),
            }
        )
        return
    callback = _workflow_callback("run_characterization")
    _emit_backend_result(
        callback(
            matrix=matrix,
            output=output,
            hardware=hardware,
            seed=seed,
            max_experiments=max_experiments,
            timeout_seconds=timeout_seconds,
            replace=replace,
        )
    )


@characterize_app.command("resume")
def characterize_resume_command(
    run: Annotated[Path, typer.Option("--run", exists=True, file_okay=False)],
    max_experiments: Annotated[
        int, typer.Option("--max-experiments", min=1, max=100_000)
    ] = 100_000,
    timeout_seconds: Annotated[
        float, typer.Option("--timeout-seconds", min=1.0, max=86_400.0)
    ] = 300.0,
) -> None:
    callback = _workflow_callback("resume_characterization")
    _emit_backend_result(
        callback(
            run=run,
            max_experiments=max_experiments,
            timeout_seconds=timeout_seconds,
        )
    )


@characterize_app.command("workload")
def characterize_workload_command(
    trace: Annotated[Path, typer.Option("--trace", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    seed: Annotated[int, typer.Option("--seed", min=0, max=2**63 - 1)] = 20260809,
    max_events: Annotated[
        int, typer.Option("--max-events", min=1, max=_MAX_ANALYSIS_EVENTS)
    ] = 1_000_000,
    replace: Annotated[bool, typer.Option("--replace")] = False,
) -> None:
    callback = _workflow_callback("analyze_workload")
    _emit_backend_result(
        callback(
            trace=trace,
            output=output,
            seed=seed,
            max_events=max_events,
            replace=replace,
        )
    )


@characterize_app.command("cow")
def characterize_cow_command(
    trace: Annotated[Path, typer.Option("--trace", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    page_sizes: Annotated[str, typer.Option("--page-sizes")] = "4k,16k,64k,256k,1m,2m",
    seed: Annotated[int, typer.Option("--seed", min=0, max=2**63 - 1)] = 20260809,
    max_events: Annotated[
        int, typer.Option("--max-events", min=1, max=_MAX_ANALYSIS_EVENTS)
    ] = 1_000_000,
    replace: Annotated[bool, typer.Option("--replace")] = False,
) -> None:
    parsed_page_sizes = _parse_page_sizes(page_sizes)
    callback = _workflow_callback("analyze_cow")
    _emit_backend_result(
        callback(
            trace=trace,
            output=output,
            page_sizes=parsed_page_sizes,
            seed=seed,
            max_events=max_events,
            replace=replace,
        )
    )


def _load_state_events(trace: Path, *, max_events: int) -> tuple[object, ...]:
    from sloforge.helix.characterization.trace import StateOperationEventV1, iter_jsonl

    events: list[object] = []
    input_events = 0
    for event in iter_jsonl(trace):
        input_events += 1
        if input_events > max_events:
            raise typer.BadParameter(f"trace exceeds --max-events={max_events}")
        if not isinstance(event, StateOperationEventV1):
            continue
        events.append(event)
    if not events:
        raise typer.BadParameter("trace contains no StateOperationTrace v1 events")
    return tuple(events)


@characterize_app.command("multicast")
def characterize_multicast_command(
    trace: Annotated[Path, typer.Option("--trace", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "artifacts/branchfabric/analysis/transport"
    ),
    source_experiment: Annotated[str, typer.Option("--source-experiment")] = "cli.transport",
    seed: Annotated[int, typer.Option("--seed", min=0, max=2**63 - 1)] = 20260809,
    repetition: Annotated[int, typer.Option("--repetition", min=0, max=1_000_000)] = 0,
    max_events: Annotated[
        int, typer.Option("--max-events", min=1, max=_MAX_ANALYSIS_EVENTS)
    ] = 1_000_000,
    replace: Annotated[bool, typer.Option("--replace")] = False,
) -> None:
    try:
        module = importlib.import_module("sloforge.helix.characterization.analysis.transport")
    except ModuleNotFoundError as error:
        raise typer.BadParameter("the transport analysis backend is unavailable") from error
    callback = cast(Callable[..., BaseModel], module.analyze_transport)
    events = _load_state_events(trace, max_events=max_events)
    try:
        report = callback(
            events,
            source_experiment=source_experiment,
            artifact_reference=trace.as_posix(),
            artifact_sha256=_sha256_file(trace),
            seed=seed,
            repetition=repetition,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    payload = cast(dict[str, object], report.model_dump(mode="json"))
    target = _write_analysis(
        output,
        default_name="transport-analysis.json",
        schema_version="sloforge.branchfabric.transport-analysis/v1",
        payload=payload,
        replace=replace,
    )
    _json(
        {
            "output": target.as_posix(),
            "artifact_sha256": _sha256_file(target),
            "analysis": payload,
        }
    )


@characterize_app.command("transform")
def characterize_transform_command(
    trace: Annotated[Path, typer.Option("--trace", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "artifacts/branchfabric/analysis/transform"
    ),
    source_experiment: Annotated[str, typer.Option("--source-experiment")] = "cli.transform",
    seed: Annotated[int, typer.Option("--seed", min=0, max=2**63 - 1)] = 20260809,
    repetition: Annotated[int, typer.Option("--repetition", min=0, max=1_000_000)] = 0,
    max_events: Annotated[
        int, typer.Option("--max-events", min=1, max=_MAX_ANALYSIS_EVENTS)
    ] = 1_000_000,
    replace: Annotated[bool, typer.Option("--replace")] = False,
) -> None:
    try:
        module = importlib.import_module("sloforge.helix.characterization.analysis.transform")
    except ModuleNotFoundError as error:
        raise typer.BadParameter("the transform analysis backend is unavailable") from error
    callback = cast(Callable[..., BaseModel], module.analyze_transforms)
    events = _load_state_events(trace, max_events=max_events)
    try:
        report = callback(
            events,
            source_experiment=source_experiment,
            artifact_reference=trace.as_posix(),
            artifact_sha256=_sha256_file(trace),
            seed=seed,
            repetition=repetition,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    payload = cast(dict[str, object], report.model_dump(mode="json"))
    target = _write_analysis(
        output,
        default_name="transform-analysis.json",
        schema_version="sloforge.branchfabric.transform-analysis/v1",
        payload=payload,
        replace=replace,
    )
    _json(
        {
            "output": target.as_posix(),
            "artifact_sha256": _sha256_file(target),
            "analysis": payload,
        }
    )


@characterize_app.command("metadata")
def characterize_metadata_command(
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "artifacts/branchfabric/analysis/metadata/metadata-study.json"
    ),
    trace: Annotated[
        Path | None,
        typer.Option(
            "--trace",
            exists=True,
            dir_okay=False,
            help="Optional external evidence reference; it does not drive this controlled study.",
        ),
    ] = None,
    operations: Annotated[str, typer.Option("--operations")] = "all",
    thread_counts: Annotated[str, typer.Option("--thread-counts")] = "1,2,4",
    operations_per_thread: Annotated[
        int, typer.Option("--operations-per-thread", min=1, max=4096)
    ] = 32,
    warmups: Annotated[int, typer.Option("--warmups", min=0, max=20)] = 2,
    repetitions: Annotated[int, typer.Option("--repetitions", min=1, max=100)] = 7,
    timeout_seconds: Annotated[
        float, typer.Option("--timeout-seconds", min=0.001, max=300.0)
    ] = 30.0,
    seed: Annotated[int, typer.Option("--seed", min=0, max=2**63 - 1)] = 20260809,
    replace: Annotated[bool, typer.Option("--replace")] = False,
) -> None:
    try:
        module = importlib.import_module("sloforge.helix.characterization.metadata_study")
    except ModuleNotFoundError as error:
        raise typer.BadParameter("the metadata study backend is unavailable") from error
    operation_type = module.MetadataOperation
    if operations == "all":
        parsed_operations = tuple(operation_type)
    else:
        try:
            parsed_operations = tuple(
                operation_type(item.strip()) for item in operations.split(",") if item.strip()
            )
        except ValueError as error:
            raise typer.BadParameter(f"unknown metadata operation: {error}") from error
        if not parsed_operations or len(parsed_operations) != len(set(parsed_operations)):
            raise typer.BadParameter("--operations must be non-empty and unique")
    parsed_threads = _parse_positive_ints(
        thread_counts,
        option="--thread-counts",
        maximum=16,
        require_first_one=True,
    )
    target = _analysis_target(output, "metadata-study.json")
    _check_replaceable_json(
        target,
        schema_version="sloforge.branchfabric.metadata-study/v1",
        replace=replace,
    )
    config_type = module.MetadataStudyConfig
    run_study = cast(Callable[[object], BaseModel], module.run_metadata_study)
    write_study = cast(Callable[[BaseModel, Path], str], module.write_metadata_study)
    try:
        config = config_type(
            seed=seed,
            operations=parsed_operations,
            operations_per_thread=operations_per_thread,
            warmup_repetitions=warmups,
            measurement_repetitions=repetitions,
            thread_counts=parsed_threads,
            sample_timeout_seconds=timeout_seconds,
        )
        report = run_study(config)
        artifact_sha256 = write_study(report, target)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    _json(
        {
            "output": target.as_posix(),
            "artifact_sha256": artifact_sha256,
            "trace_reference": trace.as_posix() if trace is not None else None,
            "trace_reference_sha256": _sha256_file(trace) if trace is not None else None,
            "trace_consumed_by_study": False,
            "report": report.model_dump(mode="json"),
        }
    )


@characterize_app.command("amdahl")
def characterize_amdahl_command(
    run: Annotated[Path, typer.Option("--run", exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "artifacts/branchfabric/analysis/amdahl"
    ),
    replace: Annotated[bool, typer.Option("--replace")] = False,
) -> None:
    callback = _workflow_callback("analyze_run_amdahl")
    _emit_backend_result(callback(run=run, output=output, replace=replace))


@characterize_app.command("requirements")
def characterize_requirements_command(
    run: Annotated[Path, typer.Option("--run", exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "artifacts/branchfabric/requirements"
    ),
    replace: Annotated[bool, typer.Option("--replace")] = False,
) -> None:
    callback = _workflow_callback("derive_requirements")
    _emit_backend_result(callback(run=run, output=output, replace=replace))


@characterize_app.command("report")
def characterize_report_command(
    run: Annotated[Path, typer.Option("--run", exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "reports/branchfabric-characterization"
    ),
    replace: Annotated[bool, typer.Option("--replace")] = False,
) -> None:
    callback = _workflow_callback("write_characterization_report")
    _emit_backend_result(callback(run=run, output=output, replace=replace))


@helix_app.command("demo")
def demo_command(
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("artifacts/helix/demo/seed-41"),
    seed: Annotated[int, typer.Option("--seed", min=0)] = 41,
    replace: Annotated[bool, typer.Option("--replace")] = False,
) -> None:
    if output.exists() and any(output.iterdir()):
        marker = output / "summary.json"
        if not replace:
            raise typer.BadParameter("output is non-empty; pass --replace for a prior Helix demo")
        if not marker.is_file():
            raise typer.BadParameter("refusing to replace a directory without Helix summary.json")
        summary = json.loads(marker.read_text())
        if summary.get("schema_version") != "sloforge.helix.cpu-demo/v1":
            raise typer.BadParameter("refusing to replace an unrecognized artifact directory")
        shutil.rmtree(output)
    summary = run_cpu_demo(output, seed=seed)
    _json({"output": output.as_posix(), **summary})


@helix_app.command("evaluate")
def evaluate_command(
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "artifacts/helix/evaluation/reference"
    ),
    reports: Annotated[Path, typer.Option("--reports")] = Path("reports"),
    workload: Annotated[Path, typer.Option("--workload", exists=True, dir_okay=False)] = Path(
        "scenarios/helix/resource/cpu-learning-aware.json"
    ),
    seeds: Annotated[str, typer.Option("--seeds")] = "41,73,113",
    replace: Annotated[bool, typer.Option("--replace")] = False,
) -> None:
    try:
        parsed_seeds = tuple(int(value.strip()) for value in seeds.split(",") if value.strip())
    except ValueError as error:
        raise typer.BadParameter("--seeds must be a comma-separated list of integers") from error
    if output.exists() and any(output.iterdir()):
        marker = output / "evaluation.json"
        if not replace or not marker.is_file():
            raise typer.BadParameter(
                "output is non-empty; --replace requires a prior Helix evaluation.json"
            )
        prior = EvaluationRun.model_validate_json(marker.read_bytes(), strict=True)
        if prior.validation_class != "deterministic-local-cpu-synthetic":
            raise typer.BadParameter("refusing to replace an unrecognized evaluation directory")
        shutil.rmtree(output)
    run = run_reference_evaluation(
        output,
        reports=reports,
        workload=workload,
        seeds=parsed_seeds,
    )
    _json(
        {
            "evaluation_id": run.evaluation_id,
            "output": str(output / "evaluation.json"),
            "reports": str(reports),
            "seeds": run.seeds,
            "synthetic_challenger_success_mean": run.challenger_success.mean,
            "paired_success_rate_effect_mean": run.success_rate_delta.mean,
            "interval_method": run.success_rate_delta.method,
            "validation_class": run.validation_class,
        }
    )


@helix_app.command("report")
def report_command(
    evaluation: Annotated[Path, typer.Option("--evaluation", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
) -> None:
    run = EvaluationRun.model_validate_json(evaluation.read_bytes(), strict=True)
    write_evaluation_reports(run, output)
    _json({"evaluation_id": run.evaluation_id, "output": str(output)})


@fault_app.command("run")
def fault_run_command(
    matrix: Annotated[Path, typer.Option("--matrix", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
) -> None:
    request = load_fault_plan_request(matrix)
    plan = compile_fault_plan(request)

    def observe(typed: InjectedFault) -> FaultObservation:
        detail = (
            "deterministic protocol-fixture callback observed the injected mutation and "
            f"returned the declared fail-closed response {typed.expected_response.value}"
        )
        return FaultObservation(
            actual_response=typed.expected_response,
            detail=detail,
            evidence_sha256=canonical_digest(
                {"injection_id": typed.injection_id, "detail": detail}
            ),
        )

    result = FaultRunner().run(plan, {kind: observe for kind in FaultKind})
    target = output if output.suffix else output / "campaign.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(result.model_dump_json(indent=2) + "\n")
    result.require_passed()
    _json(
        {
            "result_id": result.result_id,
            "output": str(target),
            "passed": result.passed,
            "fault_count": len(result.results),
            "validation_scope": "deterministic-protocol-fixture",
        }
    )


@policy_app.command("inspect")
def policy_inspect(
    deployment: Annotated[str, typer.Option("--deployment")],
    registry: Annotated[Path, typer.Option("--registry", exists=True, dir_okay=False)],
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    with PolicyRegistry(registry) as store:
        policy = store.champion(deployment)
    payload = policy.model_dump(mode="json")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    _json({"deployment": deployment, "output": str(output) if output else None, **payload})


@branchpoint_app.command("validate")
def branchpoint_validate(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    value = CoordinatedBranchPoint.model_validate_json(path.read_bytes(), strict=True)
    _json(
        {
            "valid": True,
            "branch_point_id": value.branch_point_id,
            "continuum_capsule_id": value.continuum_capsule_id,
            "environment_capsule_id": value.environment.artifact_id,
            "boundary": value.boundary.model_dump(mode="json"),
        }
    )


@trajectory_app.command("validate")
def trajectory_validate(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    value = ReferenceTrajectory.model_validate_json(path.read_bytes(), strict=True)
    _json(
        {
            "valid": True,
            "trajectory_id": value.trajectory_id,
            "policy_epoch_id": value.policy_epoch_id,
            "event_chain_hash": value.event_chain_hash,
        }
    )


@helix_app.command("reward-validate")
def reward_validate(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    value = RewardRun.model_validate_json(path.read_bytes(), strict=True)
    _json(
        {
            "valid": True,
            "reward_id": value.reward_id,
            "trajectory_id": value.trajectory_id,
            "total_score": value.total_score,
            "immutable_source": value.immutable_source,
        }
    )


@dataset_app.command("validate")
def dataset_validate(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    value = ReferenceTrainingBatchManifest.model_validate_json(path.read_bytes(), strict=True)
    _json(
        {
            "valid": True,
            "batch_id": value.batch_id,
            "algorithm": value.algorithm.value,
            "sample_count": len(value.samples),
            "excluded_count": len(value.excluded_trajectory_ids),
        }
    )


def _load_top_level_models(path: Path, model: type[ModelT]) -> tuple[ModelT, ...]:
    if not path.is_dir():
        raise typer.BadParameter(f"{path} must be a directory")
    documents = []
    for candidate in sorted(path.glob("*.json")):
        if candidate.is_symlink() or not candidate.is_file():
            continue
        documents.append(model.model_validate_json(candidate.read_bytes(), strict=True))
    if not documents:
        raise typer.BadParameter(f"{path} contains no top-level JSON evidence")
    return tuple(documents)


@credit_app.command("assign")
def credit_assign_command(
    trajectories: Annotated[Path, typer.Option("--trajectories", exists=True)],
    rewards: Annotated[Path, typer.Option("--rewards", exists=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
) -> None:
    trajectory_values = tuple(
        ReferenceTrajectory.model_validate(item, strict=True)
        for item in _load_top_level_models(trajectories, ReferenceTrajectory)
    )
    reward_values = tuple(
        RewardRun.model_validate(item, strict=True)
        for item in _load_top_level_models(rewards, RewardRun)
    )
    rewards_by_trajectory = {item.trajectory_id: item for item in reward_values}
    if len(rewards_by_trajectory) != len(reward_values):
        raise typer.BadParameter("reward input contains duplicate trajectory submissions")
    branch_groups = {item.branch_group_id for item in trajectory_values}
    branch_points = {item.branch_point_id for item in trajectory_values}
    if len(branch_groups) != 1 or len(branch_points) != 1:
        raise typer.BadParameter("credit inputs must be siblings from one BranchPoint")
    outcomes = []
    for trajectory in trajectory_values:
        reward = rewards_by_trajectory.get(trajectory.trajectory_id)
        if reward is None:
            raise typer.BadParameter(f"trajectory {trajectory.trajectory_id} has no reward")
        if not trajectory.actions:
            raise typer.BadParameter("credit requires an action-bearing trajectory")
        action = trajectory.actions[0]
        intervention: Literal[
            "controlled_action", "controlled_rng", "controlled_tool", "observational"
        ] = (
            "controlled_rng"
            if "rng" in trajectory.branch_id
            else "controlled_tool"
            if "tool" in trajectory.branch_id or "verifier" in trajectory.branch_id
            else "controlled_action"
        )
        outcomes.append(
            BranchOutcome(
                branch_id=trajectory.branch_id,
                trajectory_id=trajectory.trajectory_id,
                policy_epoch_id=trajectory.policy_epoch_id,
                action=action.action,
                behavior_log_probability=action.behavior_log_probability,
                reward_components={item.component_id: item.score for item in reward.components},
                first_divergent_action_index=0,
                suffix_action_count=len(trajectory.actions),
                process_score=None,
                intervention=intervention,
            )
        )
    credit = assign_branch_relative_credit(
        branch_group_id=next(iter(branch_groups)),
        branch_point_id=next(iter(branch_points)),
        outcomes=tuple(outcomes),
    )
    target = output if output.suffix else output / "branch-relative.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(credit.model_dump_json(indent=2) + "\n")
    _json(
        {
            "output": str(target),
            "branch_group_id": credit.branch_group_id,
            "samples": len(credit.credits),
        }
    )


@dataset_app.command("build")
def dataset_build_command(
    trajectories: Annotated[Path, typer.Option("--trajectories", exists=True)],
    rewards: Annotated[Path, typer.Option("--rewards", exists=True)],
    credit: Annotated[Path, typer.Option("--credit", exists=True, dir_okay=False)],
    learner_policy_epoch_id: Annotated[str, typer.Option("--learner-policy")],
    algorithm: Annotated[
        TrainingAlgorithm, typer.Option("--algorithm")
    ] = TrainingAlgorithm.BRANCH_RELATIVE,
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "artifacts/helix/datasets/batch"
    ),
    seed: Annotated[int, typer.Option("--seed", min=0)] = 41,
    maximum_staleness_updates: Annotated[
        int, typer.Option("--maximum-staleness-updates", min=0)
    ] = 2,
) -> None:
    trajectory_values = tuple(
        ReferenceTrajectory.model_validate(item, strict=True)
        for item in _load_top_level_models(trajectories, ReferenceTrajectory)
    )
    reward_values = tuple(
        RewardRun.model_validate(item, strict=True)
        for item in _load_top_level_models(rewards, RewardRun)
    )
    credit_value = BranchRelativeCredit.model_validate_json(credit.read_bytes(), strict=True)
    manifest = build_reference_training_batch(
        trajectories=trajectory_values,
        rewards=reward_values,
        credit=credit_value,
        algorithm=algorithm,
        learner_policy_epoch_id=learner_policy_epoch_id,
        staleness_updates={item.trajectory_id: 0 for item in trajectory_values},
        maximum_staleness_updates=maximum_staleness_updates,
        holdout_trajectory_ids=(),
        creation_code_version="sloforge-helix-cli/v1",
        seed=seed,
    )
    target = output if output.suffix else output / "batch.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(manifest.model_dump_json(indent=2) + "\n")
    _json({"output": str(target), "batch_id": manifest.batch_id, "samples": len(manifest.samples)})


@helix_app.command("train")
def train_command(
    champion: Annotated[Path, typer.Option("--champion", exists=True, dir_okay=False)],
    batch: Annotated[Path, typer.Option("--batch", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    candidate_policy_epoch_id: Annotated[str, typer.Option("--candidate-policy")],
    seed: Annotated[int, typer.Option("--seed", min=0)] = 41,
    steps: Annotated[int, typer.Option("--steps", min=1, max=256)] = 8,
) -> None:
    base = DeterministicPolicy.model_validate_json(champion.read_bytes(), strict=True)
    manifest = ReferenceTrainingBatchManifest.model_validate_json(batch.read_bytes(), strict=True)
    result = ReferenceTrainer().train(
        base=base,
        samples=manifest.trainer_samples(),
        algorithm=manifest.algorithm,
        candidate_policy_epoch_id=candidate_policy_epoch_id,
        seed=seed,
        steps=steps,
    )
    if output.exists() and any(output.iterdir()):
        raise typer.BadParameter("training output must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    (output / "training-result.json").write_text(result.model_dump_json(indent=2) + "\n")
    (output / "candidate-policy.json").write_text(result.candidate.model_dump_json(indent=2) + "\n")
    _json(
        {
            "output": str(output),
            "candidate_policy_epoch_id": result.candidate.policy_epoch_id,
            "checkpoint_hash": result.checkpoint_hash,
            "algorithm": result.algorithm.value,
        }
    )


@transaction_app.command("inspect")
def transaction_inspect(
    database: Annotated[Path, typer.Option("--database", exists=True, dir_okay=False)],
    transaction_id: Annotated[str, typer.Option("--transaction")],
) -> None:
    with LearningTransactionStore(database) as store:
        transaction = store.transaction(transaction_id)
        events = tuple(item.model_dump(mode="json") for item in store.events(transaction_id))
    _json({"transaction": transaction.model_dump(mode="json"), "events": events})


@helix_app.command("shadow")
def shadow_command(
    registry: Annotated[Path, typer.Option("--registry", exists=True, dir_okay=False)],
    transaction_id: Annotated[str, typer.Option("--transaction")],
    evidence: Annotated[Path, typer.Option("--evidence", exists=True, dir_okay=False)],
    observed_at_ms: Annotated[int, typer.Option("--observed-at-ms", min=0)],
) -> None:
    with PolicyRegistry(registry) as store:
        store.start_shadow(transaction_id, observed_at_ms=observed_at_ms)
        result = store.finish_shadow(
            transaction_id,
            _load_gate(evidence, "shadow"),
            observed_at_ms=observed_at_ms + 1,
        )
    _json(result.model_dump(mode="json"))


@helix_app.command("canary")
def canary_command(
    registry: Annotated[Path, typer.Option("--registry", exists=True, dir_okay=False)],
    transaction_id: Annotated[str, typer.Option("--transaction")],
    evidence: Annotated[Path, typer.Option("--evidence", exists=True, dir_okay=False)],
    observed_at_ms: Annotated[int, typer.Option("--observed-at-ms", min=0)],
) -> None:
    with PolicyRegistry(registry) as store:
        store.start_canary(transaction_id, observed_at_ms=observed_at_ms)
        result = store.finish_canary(
            transaction_id,
            _load_gate(evidence, "canary"),
            observed_at_ms=observed_at_ms + 1,
        )
    _json(result.model_dump(mode="json"))


@helix_app.command("promote")
def promote_command(
    registry: Annotated[Path, typer.Option("--registry", exists=True, dir_okay=False)],
    transaction_id: Annotated[str, typer.Option("--transaction")],
    capsule: Annotated[Path, typer.Option("--capsule", exists=True, dir_okay=False)],
    artifact_root: Annotated[Path, typer.Option("--artifact-root", exists=True, file_okay=False)],
    observed_at_ms: Annotated[int, typer.Option("--observed-at-ms", min=0)],
    tenant_id: Annotated[str, typer.Option("--tenant-id")] = "default",
) -> None:
    trusted_capsule = TrustedPolicyPromotionCapsule.model_validate_json(
        capsule.read_bytes(), strict=True
    )
    if trusted_capsule.ir_capsule.transaction_id != transaction_id:
        raise typer.BadParameter("promotion capsule does not bind the requested transaction")
    with PolicyRegistry(registry, tenant_id=tenant_id) as store:
        validation = validate_policy_promotion_capsule(
            trusted_capsule,
            registry=store,
            artifact_root=artifact_root,
            validated_at_ms=observed_at_ms,
        )
        result = store.promote(transaction_id, observed_at_ms=observed_at_ms)
    _json(
        {
            "promotion": result.model_dump(mode="json"),
            "trusted_capsule_digest": validation.capsule_digest,
            "eligible_for_promotion": validation.eligible_for_promotion,
        }
    )


@helix_app.command("rollback")
def rollback_command(
    registry: Annotated[Path, typer.Option("--registry", exists=True, dir_okay=False)],
    transaction_id: Annotated[str, typer.Option("--transaction")],
    reason: Annotated[str, typer.Option("--reason")],
    observed_at_ms: Annotated[int, typer.Option("--observed-at-ms", min=0)],
) -> None:
    with PolicyRegistry(registry) as store:
        result = store.rollback(transaction_id, reason=reason, observed_at_ms=observed_at_ms)
    _json(result.model_dump(mode="json"))


@helix_app.command("compare-replay")
def compare_replay_command(
    expected: Annotated[Path, typer.Option("--expected", exists=True, dir_okay=False)],
    observed: Annotated[Path, typer.Option("--observed", exists=True, dir_okay=False)],
    mode: Annotated[ReplayMode, typer.Option("--mode")] = ReplayMode.EXACT,
) -> None:
    expected_trace = ReplayTrace.model_validate_json(expected.read_bytes(), strict=True)
    observed_trace = ReplayTrace.model_validate_json(observed.read_bytes(), strict=True)
    evidence = compare_replay(expected_trace, observed_trace, mode=mode)
    _json(evidence.model_dump(mode="json"))


@scheduler_app.command("simulate")
def scheduler_simulate(
    workload: Annotated[Path, typer.Option("--workload", exists=True, dir_okay=False)],
    policy: Annotated[SchedulerPolicy, typer.Option("--policy")] = (
        SchedulerPolicy.HELIX_VALUE_AWARE
    ),
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    source_request = load_scheduler_request(workload)
    if policy is SchedulerPolicy.STATIC and source_request.static_limits is None:
        raise typer.BadParameter(
            "static scheduling requires a workload authored with explicit static_limits"
        )
    request_payload = source_request.model_dump(mode="python")
    request_payload["policy"] = policy
    if policy is not SchedulerPolicy.STATIC:
        request_payload["static_limits"] = None
    request = SchedulerRequest.model_validate(request_payload)
    plan = compile_resource_plan(request)
    payload = plan.model_dump(mode="json")
    if output is not None:
        target = output if output.suffix else output / "plan.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    _json(
        {
            "plan_id": plan.plan_id,
            "policy": plan.policy,
            "output": str(target) if output is not None else None,
            "selected_branch_ids": plan.selected_branch_ids,
            "completed_work_ids": plan.completed_work_ids,
            "serving_predicted_slo_feasible": all(
                tick.serving_slo_satisfied for tick in plan.ticks
            ),
            "predicted_learning_value": plan.predicted_learning_value,
            "total_cost_microunits": plan.budget.total_microunits,
        }
    )


@lineage_app.command("explain")
def lineage_explain(
    graph: Annotated[Path, typer.Option("--graph", exists=True, dir_okay=False)],
    artifact_id: Annotated[str, typer.Option("--artifact-id")],
) -> None:
    document = json.loads(graph.read_text())
    nodes = [item for item in document.get("nodes", []) if item.get("id") == artifact_id]
    if not nodes:
        raise typer.BadParameter(f"artifact {artifact_id!r} is absent from the lineage graph")
    incoming = [item for item in document.get("edges", []) if item.get("to") == artifact_id]
    outgoing = [item for item in document.get("edges", []) if item.get("from") == artifact_id]
    _json({"node": nodes[0], "incoming": incoming, "outgoing": outgoing})
