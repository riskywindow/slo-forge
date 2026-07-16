"""Scriptable SLOForge Continuum commands."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Literal

import typer
import yaml

from sloforge.continuum.adapters import ReferenceHeadMajorAdapter, ReferenceTokenMajorAdapter
from sloforge.continuum.adapters.genesis import probe_genesis
from sloforge.continuum.adapters.pytorch import probe_pytorch
from sloforge.continuum.adapters.sglang import probe_sglang
from sloforge.continuum.adapters.vllm import probe_vllm
from sloforge.continuum.benchmarking import (
    EvaluationRequest,
    load_evaluation,
    run_evaluation_campaign,
)
from sloforge.continuum.conversion import KVLayout, KVLayoutKind, compile_conversion
from sloforge.continuum.demo import (
    FlagshipDemoRequest,
    FlagshipDemoResult,
    run_flagship_demo,
    write_flagship_artifact,
)
from sloforge.continuum.ir import (
    CompressionKind,
    EncryptionKind,
    ExecutionStateCapsule,
    load_capsule,
    save_document,
    validate_capsule,
)
from sloforge.continuum.operations import (
    CheckpointArtifact,
    checkpoint_full,
    clone_checkpoint,
    load_checkpoint_artifact,
    pause_and_checkpoint,
    resume_checkpoint,
    save_checkpoint_artifact,
)
from sloforge.continuum.planner import MigrationPlanningInput, plan_migration
from sloforge.continuum.reports import generate_reports
from sloforge.continuum.storage import ChunkRef, FileContentStore
from sloforge.continuum.transaction import (
    CutoverPhase,
    DurableCoordinator,
    GatewayCommitLedger,
    SessionLease,
)
from sloforge.continuum.transaction import TokenEvent as GatewayTokenEvent
from sloforge.util import git_commit, sha256_file, write_json

continuum_app = typer.Typer(
    help="Capture, validate, convert, and transactionally migrate active execution state.",
    no_args_is_help=True,
)
runtime_app = typer.Typer(help="Inspect version-scoped Continuum runtime capabilities.")
state_app = typer.Typer(help="Inspect and capture logical and physical execution state.")
capsule_app = typer.Typer(help="Validate proof-carrying execution state capsules.")
migration_app = typer.Typer(help="Run and inspect transactional state migrations.")
conversion_app = typer.Typer(help="Compile bounded direct physical-state conversions.")
continuum_app.add_typer(runtime_app, name="runtime")
continuum_app.add_typer(state_app, name="state")
continuum_app.add_typer(capsule_app, name="capsule")
continuum_app.add_typer(migration_app, name="migration")
continuum_app.add_typer(conversion_app, name="conversion")


def _prepare_output(output: Path, *, reset: bool) -> None:
    if output.exists():
        if not reset:
            raise typer.BadParameter(f"output already exists: {output}; pass --reset to replace it")
        resolved = output.resolve()
        repository = Path(__file__).resolve().parents[3]
        if resolved in {repository, repository.parent, Path("/")}:
            raise typer.BadParameter("refusing to reset a broad directory")
        if output.is_dir():
            shutil.rmtree(output)
        else:
            output.unlink()
    output.mkdir(parents=True)


def _write_output(path: Path, value: object) -> None:
    write_json(path, value)
    typer.echo(str(path))


@runtime_app.command("inspect")
def runtime_inspect(
    runtime: Annotated[
        Literal["reference-a", "reference-b", "pytorch", "genesis", "vllm", "sglang"],
        typer.Option("--runtime"),
    ],
    output: Annotated[Path, typer.Option("--output", "-o")],
) -> None:
    """Inspect local packages and reference adapters without mutating a live session."""

    if runtime in {"reference-a", "reference-b"}:
        adapter = (
            ReferenceTokenMajorAdapter()
            if runtime == "reference-a"
            else ReferenceHeadMajorAdapter()
        )
        document: object = {
            "schema_version": "sloforge.continuum.runtime-inspection/v1",
            "runtime": asdict(adapter.identity),
            "capabilities": {
                "operations": sorted(item.value for item in adapter.capabilities.operations),
                "state_types": sorted(item.value for item in adapter.capabilities.state_types),
                "layouts": [asdict(item) for item in adapter.capabilities.layouts],
                "dirty_tracking": sorted(
                    item.value for item in adapter.capabilities.dirty_tracking_strategies
                ),
            },
            "exercised": True,
        }
    else:
        probe = {
            "pytorch": probe_pytorch,
            "genesis": probe_genesis,
            "vllm": probe_vllm,
            "sglang": probe_sglang,
        }[runtime]()
        document = {
            "schema_version": "sloforge.continuum.runtime-inspection/v1",
            **asdict(probe),
        }
    _write_output(output, document)


@state_app.command("capture")
def state_capture(
    output: Annotated[Path, typer.Option("--output", "-o")],
    session: Annotated[str, typer.Option("--session")],
    seed: Annotated[int, typer.Option("--seed")],
    runtime_inspection: Annotated[
        Path | None,
        typer.Option("--runtime", exists=True, dir_okay=False),
    ] = None,
    generated_tokens: Annotated[int, typer.Option("--generated-tokens", min=1)] = 8,
    input_tokens: Annotated[str, typer.Option("--input-tokens")] = "2,3,5,7",
) -> None:
    """Create a deterministic live reference session and publish a consistent capsule."""

    try:
        parsed_input = tuple(int(item) for item in input_tokens.split(",") if item)
    except ValueError as error:
        raise typer.BadParameter("input tokens must be comma-separated integers") from error
    if runtime_inspection is not None:
        inspection = json.loads(runtime_inspection.read_text(encoding="utf-8"))
        inspected_name = inspection.get("runtime", {}).get("runtime_name")
        if inspected_name != ReferenceTokenMajorAdapter().identity.runtime_name:
            raise typer.BadParameter(
                "state capture currently requires a reference-a runtime inspection"
            )
    runtime = ReferenceTokenMajorAdapter()
    runtime.create_session(
        session_id=session,
        request_id=f"request-{session}",
        tenant_id="local-continuum",
        input_token_ids=parsed_input,
        seed=seed,
    )
    for event in runtime.stream_tokens(session, count=generated_tokens):
        runtime.acknowledge_gateway(
            session,
            token_index=event.token_index,
            owner_epoch=event.owner_epoch,
        )
    captured = runtime.capture_consistent(session)
    output.mkdir(parents=True, exist_ok=True)
    lease = SessionLease(
        session_id=session,
        owner_runtime=runtime.identity.runtime_name,
        owner_epoch=captured.logical.owner_epoch,
        fencing_token=captured.logical.owner_epoch,
        expiration_ms=60_000,
        coordinator_version=captured.logical.owner_epoch,
        last_committed_state_version=captured.logical.state_version,
        last_committed_token_index=generated_tokens - 1,
    )
    with FileContentStore(output / "store") as store:
        checkpoint = checkpoint_full(
            runtime,
            session,
            store=store,
            lease=lease,
            published_at_ms=0,
            capture_timestamp="deterministic-reference-capture",
            git_commit=git_commit(Path(__file__).resolve().parents[3]),
            continuum_version="0.1.0",
        )
        save_document(output / "capsule.json", checkpoint.capsule)
        save_checkpoint_artifact(output / "checkpoint.json", checkpoint)
        _write_output(
            output / "capture-result.json",
            {
                "schema_version": "sloforge.continuum.capture-result/v1",
                "session_id": session,
                "capsule_id": checkpoint.capsule.identity.capsule_id,
                "capsule": str(output / "capsule.json"),
                "checkpoint": str(output / "checkpoint.json"),
                "store_manifest_id": checkpoint.store_manifest.manifest_id,
                "segment_count": len(captured.segments),
                "state_version": captured.logical.state_version,
                "seed": seed,
                "transaction_id": None,
                "transaction_phase": None,
                "lease_scope": "ephemeral_capture_binding_not_persisted",
                "durable_lease_retained": False,
            },
        )


@state_app.command("inspect")
def state_inspect(
    capsule: Annotated[Path, typer.Option("--capsule", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
) -> None:
    """Inspect logical components separately from their physical placement."""

    document = load_capsule(capsule)
    _write_output(
        output,
        {
            "schema_version": "sloforge.continuum.state-inspection/v1",
            "session_id": document.identity.session_id,
            "owner_epoch": document.identity.owner_epoch,
            "logical_components": [
                component.semantic_id
                for component in document.logical_state.component_descriptors()
            ],
            "dependency_edges": len(document.logical_state.dependency_graph.edges),
            "runtime": document.physical_state.runtime.model_dump(mode="json"),
            "segments": len(document.physical_state.segments),
            "layouts": [
                layout.model_dump(mode="json")
                for layout in document.physical_state.layout_descriptors
            ],
            "non_portable_runtime_state": list(
                document.physical_state.reconstructible_runtime_state
            ),
        },
    )


@capsule_app.command("validate")
def capsule_validate(
    capsule: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    store: Annotated[Path | None, typer.Option("--store", file_okay=False)] = None,
) -> None:
    document = load_capsule(capsule)
    validate_capsule(document)
    store_path = store if store is not None else capsule.parent / "store"
    chunks_verified = _validate_external_capsule_chunks(document, store_path)
    typer.echo(
        json.dumps(
            {
                "valid": True,
                "capsule_id": document.identity.capsule_id,
                "session_id": document.identity.session_id,
                "owner_epoch": document.identity.owner_epoch,
                "segments": len(document.physical_state.segments),
                "external_chunks_verified": chunks_verified,
                "store": str(store_path),
            },
            sort_keys=True,
        )
    )


def _validate_external_capsule_chunks(capsule: ExecutionStateCapsule, store_path: Path) -> int:
    """Resolve every external reference in the tenant store and verify segment seals."""

    # This helper is deliberately version-scoped to the uncompressed, unencrypted local CAS.
    # Transformed chunks require their typed encryption wrapper and cannot silently fall back.
    segment_manifests = capsule.segment_manifests
    if not store_path.is_dir():
        raise RuntimeError(f"capsule external content store is missing: {store_path}")
    verified = 0
    with FileContentStore(store_path) as content_store:
        for manifest in segment_manifests:
            payloads: list[bytes] = []
            for chunk in manifest.chunks:
                tenant = capsule.identity.tenant_id
                if chunk.tenant_security_domain != tenant:
                    raise RuntimeError("capsule chunk crosses its authenticated tenant domain")
                if chunk.storage_uri != f"cas://{tenant}/{chunk.content_hash.value}":
                    raise RuntimeError("capsule chunk URI is not the authenticated local CAS URI")
                if (
                    chunk.compression is not CompressionKind.NONE
                    or chunk.encryption is not EncryptionKind.NONE
                ):
                    raise RuntimeError(
                        "capsule validation refuses a silent transformed-chunk fallback"
                    )
                reference = ChunkRef(
                    tenant_id=tenant,
                    digest=chunk.content_hash.value,
                    size_bytes=chunk.size_bytes,
                    stored_bytes=chunk.size_bytes,
                    compression="none",
                )
                payload = content_store.read(tenant, reference)
                if sha256(payload).hexdigest() != chunk.content_hash.value:
                    raise RuntimeError("capsule external chunk digest verification failed")
                payloads.append(payload)
                verified += 1
            if sha256(b"".join(payloads)).hexdigest() != manifest.segment_hash.value:
                raise RuntimeError("capsule segment digest verification failed")
    return verified


def _new_reference_session(
    *, session_id: str, seed: int, generated_tokens: int
) -> ReferenceTokenMajorAdapter:
    runtime = ReferenceTokenMajorAdapter()
    runtime.create_session(
        session_id=session_id,
        request_id=f"request-{session_id}",
        tenant_id="local-continuum",
        input_token_ids=(2, 3, 5, 7),
        seed=seed,
    )
    for event in runtime.stream_tokens(session_id, count=generated_tokens):
        runtime.acknowledge_gateway(
            session_id,
            token_index=event.token_index,
            owner_epoch=event.owner_epoch,
        )
    return runtime


def _operation_lease(runtime: ReferenceTokenMajorAdapter, session_id: str) -> SessionLease:
    metadata = runtime.inspect_session(session_id)
    return SessionLease(
        session_id=session_id,
        owner_runtime=runtime.identity.runtime_name,
        owner_epoch=metadata.owner_epoch,
        fencing_token=metadata.owner_epoch,
        expiration_ms=120_000,
        coordinator_version=metadata.owner_epoch,
        last_committed_state_version=metadata.state_version,
        last_committed_token_index=metadata.client_visible_index,
    )


def _save_operation(
    output: Path,
    *,
    operation: str,
    artifact: CheckpointArtifact,
    store_path: Path,
    seed: int,
) -> None:
    checkpoint_path = output / "checkpoint.json"
    capsule_path = output / "capsule.json"
    save_checkpoint_artifact(checkpoint_path, artifact)
    save_document(capsule_path, artifact.capsule)
    _write_output(
        output / "operation-result.json",
        {
            "schema_version": "sloforge.continuum.operation-result/v1",
            "operation": operation,
            "seed": seed,
            "session_id": artifact.capsule.identity.session_id,
            "capsule_id": artifact.capsule.identity.capsule_id,
            "checkpoint": str(checkpoint_path),
            "capsule": str(capsule_path),
            "store": str(store_path),
            "transaction_terminal": True,
            "durable_lease_retained": False,
        },
    )


def _create_checkpoint_operation(
    *,
    output: Path,
    seed: int,
    session: str,
    generated_tokens: int,
    pause: bool,
    reset: bool,
) -> None:
    _prepare_output(output, reset=reset)
    runtime = _new_reference_session(
        session_id=session,
        seed=seed,
        generated_tokens=generated_tokens,
    )
    store_path = output / "store"
    with FileContentStore(store_path) as content_store:
        if pause:
            artifact = pause_and_checkpoint(
                runtime,
                session,
                store=content_store,
                lease=_operation_lease(runtime, session),
                published_at_ms=0,
                capture_timestamp="deterministic-cli-pause",
                git_commit=git_commit(Path(__file__).resolve().parents[3]),
                continuum_version="0.1.0",
            ).checkpoint
        else:
            artifact = checkpoint_full(
                runtime,
                session,
                store=content_store,
                lease=_operation_lease(runtime, session),
                published_at_ms=0,
                capture_timestamp="deterministic-cli-checkpoint",
                git_commit=git_commit(Path(__file__).resolve().parents[3]),
                continuum_version="0.1.0",
            )
    _save_operation(
        output,
        operation="pause" if pause else "checkpoint",
        artifact=artifact,
        store_path=store_path,
        seed=seed,
    )


@continuum_app.command("checkpoint")
def checkpoint_command(
    output: Annotated[Path, typer.Option("--output", "-o")],
    seed: Annotated[int, typer.Option("--seed")],
    session: Annotated[str, typer.Option("--session")] = "checkpoint-session",
    generated_tokens: Annotated[int, typer.Option("--generated-tokens", min=1)] = 8,
    reset: Annotated[bool, typer.Option("--reset")] = False,
) -> None:
    """Publish a full deterministic reference checkpoint without retaining a lease."""

    _create_checkpoint_operation(
        output=output,
        seed=seed,
        session=session,
        generated_tokens=generated_tokens,
        pause=False,
        reset=reset,
    )


@continuum_app.command("pause")
def pause_command(
    output: Annotated[Path, typer.Option("--output", "-o")],
    seed: Annotated[int, typer.Option("--seed")],
    session: Annotated[str, typer.Option("--session")] = "pause-session",
    generated_tokens: Annotated[int, typer.Option("--generated-tokens", min=1)] = 8,
    reset: Annotated[bool, typer.Option("--reset")] = False,
) -> None:
    """Pause at a token boundary and publish a deterministic reference checkpoint."""

    _create_checkpoint_operation(
        output=output,
        seed=seed,
        session=session,
        generated_tokens=generated_tokens,
        pause=True,
        reset=reset,
    )


@continuum_app.command("resume")
def resume_command(
    checkpoint: Annotated[Path, typer.Option("--checkpoint", exists=True, dir_okay=False)],
    store: Annotated[Path, typer.Option("--store", exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    seed: Annotated[int, typer.Option("--seed")],
    generated_tokens: Annotated[int, typer.Option("--generated-tokens", min=1)] = 3,
) -> None:
    """Transactionally resume an epoch-1 reference checkpoint on physical layout B."""

    artifact = load_checkpoint_artifact(checkpoint)
    if artifact.capsule.identity.owner_epoch != 1:
        raise typer.BadParameter("reference resume CLI currently requires an epoch-1 checkpoint")
    expected_model = ReferenceTokenMajorAdapter().config.model
    destination = ReferenceHeadMajorAdapter()
    session_id = artifact.capsule.identity.session_id
    watermark = artifact.capsule.transaction.commit_watermark
    accepted: list[int] = []
    with (
        FileContentStore(store) as content_store,
        DurableCoordinator(":memory:") as coordinator,
        GatewayCommitLedger(":memory:") as gateway,
    ):
        coordinator.create_lease(
            session_id=session_id,
            owner_runtime=artifact.capsule.physical_state.runtime.runtime_name,
            expiration_ms=120_000,
            initial_token_index=watermark,
        )
        gateway.register(session_id=session_id, owner_epoch=1, next_token_index=watermark + 1)
        resumed = resume_checkpoint(
            artifact,
            store=content_store,
            destination=destination,
            source_release_confirmed=True,
            expected_tenant_id=artifact.capsule.identity.tenant_id,
            expected_model=expected_model,
            coordinator=coordinator,
            gateway=gateway,
            seed=seed,
            now_ms=0,
        )
        for event in destination.stream_tokens(
            session_id,
            count=generated_tokens,
            transaction_id=resumed.transaction.transaction_id,
        ):
            gateway.accept(
                GatewayTokenEvent(
                    session_id=event.session_id,
                    owner_epoch=event.owner_epoch,
                    token_index=event.token_index,
                    token_id=event.token_id,
                    state_commit_version=event.state_commit_version,
                    transaction_id=event.transaction_id,
                )
            )
            destination.acknowledge_gateway(
                session_id,
                token_index=event.token_index,
                owner_epoch=event.owner_epoch,
            )
            accepted.append(event.token_index)
    _write_output(
        output,
        {
            "schema_version": "sloforge.continuum.resume-result/v1",
            "session_id": session_id,
            "transaction_id": resumed.transaction.transaction_id,
            "phase": resumed.transaction.phase.value,
            "source_owner_epoch": resumed.source_owner_epoch,
            "destination_owner_epoch": resumed.destination_owner_epoch,
            "accepted_token_indices": accepted,
            "transaction_terminal": resumed.transaction.phase is CutoverPhase.COMPLETED,
            "coordinator_scope": "ephemeral_local_closed",
            "durable_lease_retained": False,
        },
    )


@continuum_app.command("clone")
def clone_command(
    checkpoint: Annotated[Path, typer.Option("--checkpoint", exists=True, dir_okay=False)],
    store: Annotated[Path, typer.Option("--store", exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    session: Annotated[str, typer.Option("--session")],
    seed: Annotated[int, typer.Option("--seed")],
    reset: Annotated[bool, typer.Option("--reset")] = False,
) -> None:
    """Clone a reference checkpoint into an independent authenticated content store."""

    _prepare_output(output, reset=reset)
    parent = load_checkpoint_artifact(checkpoint)
    expected_model = ReferenceTokenMajorAdapter().config.model
    destination_store_path = output / "store"
    watermark = parent.capsule.transaction.commit_watermark
    clone_lease = SessionLease(
        session_id=session,
        owner_runtime=parent.capsule.physical_state.runtime.runtime_name,
        owner_epoch=1,
        fencing_token=1,
        expiration_ms=120_000,
        coordinator_version=1,
        last_committed_state_version=(
            parent.capsule.transaction.ownership_lease.last_committed_state_version
        ),
        last_committed_token_index=watermark,
    )
    with (
        FileContentStore(store) as source_store,
        FileContentStore(destination_store_path) as destination_store,
    ):
        cloned = clone_checkpoint(
            parent,
            source_store=source_store,
            destination_store=destination_store,
            expected_tenant_id=parent.capsule.identity.tenant_id,
            expected_model=expected_model,
            clone_lease=clone_lease,
            seed=seed,
            published_at_ms=0,
            capture_timestamp="deterministic-cli-clone",
            git_commit=git_commit(Path(__file__).resolve().parents[3]),
            continuum_version="0.1.0",
        )
    _save_operation(
        output,
        operation="clone",
        artifact=cloned.clone,
        store_path=destination_store_path,
        seed=seed,
    )


@migration_app.command("status")
def migration_status(
    transaction: Annotated[str, typer.Option("--transaction")],
    coordinator_path: Annotated[Path, typer.Option("--coordinator", exists=True, dir_okay=False)],
) -> None:
    with DurableCoordinator(coordinator_path) as coordinator:
        record = coordinator.transaction(transaction)
        journal = coordinator.journal(transaction)
    typer.echo(
        json.dumps(
            {
                "transaction": record.model_dump(mode="json"),
                "journal": [entry.model_dump(mode="json") for entry in journal],
            },
            sort_keys=True,
        )
    )


@migration_app.command("plan")
def migration_plan_command(
    request: Annotated[Path, typer.Option("--request", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
) -> None:
    """Select a migration strategy from a strict, evidence-bound planning request."""

    planning_input = MigrationPlanningInput.model_validate_json(
        request.read_text(encoding="utf-8"), strict=True
    )
    _write_output(output, plan_migration(planning_input).model_dump(mode="json"))


@conversion_app.command("compile")
def conversion_compile_command(
    output: Annotated[Path, typer.Option("--output", "-o")],
    tokens: Annotated[int, typer.Option("--tokens", min=0)],
    layers: Annotated[int, typer.Option("--layers", min=1)] = 4,
    kv_heads: Annotated[int, typer.Option("--kv-heads", min=1)] = 4,
    head_dimension: Annotated[int, typer.Option("--head-dimension", min=1)] = 8,
    source_tp: Annotated[int, typer.Option("--source-tp", min=1)] = 4,
    destination_tp: Annotated[int, typer.Option("--destination-tp", min=1)] = 2,
    source_page_size: Annotated[int, typer.Option("--source-page-size", min=1)] = 3,
    destination_page_size: Annotated[int, typer.Option("--destination-page-size", min=1)] = 5,
    maximum_temporary_bytes: Annotated[int, typer.Option("--maximum-temporary-bytes", min=1)] = 64
    * 1024,
) -> None:
    """Compile token-major/separate TP state directly to head-major/packed TP state."""

    source = KVLayout(
        kind=KVLayoutKind.TOKEN_MAJOR_SEPARATE,
        tensor_parallel_degree=source_tp,
        page_size_tokens=source_page_size,
        layer_count=layers,
        token_count=tokens,
        kv_head_count=kv_heads,
        head_dim=head_dimension,
        dtype="float32",
    )
    destination = KVLayout(
        kind=KVLayoutKind.HEAD_MAJOR_PACKED,
        tensor_parallel_degree=destination_tp,
        page_size_tokens=destination_page_size,
        layer_count=layers,
        token_count=tokens,
        kv_head_count=kv_heads,
        head_dim=head_dimension,
        dtype="float32",
    )
    _write_output(
        output,
        compile_conversion(
            source,
            destination,
            maximum_temporary_bytes=maximum_temporary_bytes,
        ).model_dump(mode="json"),
    )


@migration_app.command("modelcheck")
def migration_modelcheck(
    output: Annotated[Path, typer.Option("--output", "-o")],
    seed: Annotated[int, typer.Option("--seed")],
) -> None:
    """Run the bounded Rust protocol model checker with a strict timeout."""

    repository = Path(__file__).resolve().parents[3]
    environment = {
        key: value
        for key in ("PATH", "HOME", "CARGO_HOME", "RUSTUP_HOME", "TMPDIR")
        if (value := os.environ.get(key)) is not None
    }
    command = [
        "cargo",
        "run",
        "-q",
        "-p",
        "sloforge-state-modelcheck",
        "--",
        "--safe",
        str(seed),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=repository,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=60.0,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("bounded Continuum model checking failed") from error
    document = json.loads(completed.stdout)
    _write_output(output, document)


@continuum_app.command("migrate")
def migrate_command(
    output: Annotated[Path, typer.Option("--output", "-o")],
    seed: Annotated[int, typer.Option("--seed")] = 317,
    session: Annotated[str, typer.Option("--session")] = "session-flagship-001",
    mode: Annotated[Literal["pre-copy"], typer.Option("--mode")] = "pre-copy",
    reset: Annotated[bool, typer.Option("--reset")] = False,
) -> None:
    """Run the real CPU cross-layout migration with rollback and retry."""

    del mode
    _prepare_output(output, reset=reset)
    result = run_flagship_demo(
        FlagshipDemoRequest(
            work_dir=output / "work",
            session_id=session,
            tenant_id="local-continuum-demo",
            seed=seed,
            git_commit=git_commit(Path(__file__).resolve().parents[3]),
        )
    )
    artifact = output / "flagship.json"
    write_flagship_artifact(result, artifact)
    _write_output(
        output / "manifest.json",
        {
            "schema_version": "sloforge.continuum.demo-manifest/v1",
            "artifact": artifact.name,
            "sha256": sha256_file(artifact),
            "run_id": result.run_id,
            "seed": seed,
            "accepted_token_count": len(result.accepted_token_indices),
            "failed_transaction": result.failed_migration.transaction_id,
            "successful_transaction": result.successful_migration.transaction_id,
            "synthetic_hardware": True,
        },
    )
    shutil.rmtree(output / "work")


@migration_app.command("verify")
def migration_verify(
    artifact: Annotated[Path, typer.Option("--artifact", exists=True, dir_okay=False)],
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    """Independently validate a flagship transaction evidence bundle."""

    result = FlagshipDemoResult.model_validate_json(artifact.read_text(encoding="utf-8"))
    invariants = _verify_flagship_evidence(result, artifact=artifact)
    accepted = result.accepted_token_indices
    evidence = {
        "schema_version": "sloforge.continuum.migration-verification/v1",
        "valid": True,
        "run_id": result.run_id,
        "artifact_sha256": sha256_file(artifact),
        "transaction_id": result.successful_migration.transaction_id,
        "owner_epoch": result.successful_migration.destination_owner_epoch,
        "accepted_token_indices": accepted,
        "invariants": invariants,
        "scope": "deterministic CPU reference adapters and gateway acceptance ledger",
    }
    if output is None:
        typer.echo(json.dumps(evidence, sort_keys=True))
    else:
        _write_output(output, evidence)


_SUCCESS_PHASES = tuple(
    phase.value
    for phase in (
        CutoverPhase.PROPOSED,
        CutoverPhase.COMPATIBILITY_VALIDATED,
        CutoverPhase.DESTINATION_PREPARING,
        CutoverPhase.PRECOPYING,
        CutoverPhase.DELTA_SYNCING,
        CutoverPhase.CUTOVER_REQUESTED,
        CutoverPhase.SOURCE_QUIESCING,
        CutoverPhase.SOURCE_FROZEN,
        CutoverPhase.FINAL_DELTA_TRANSFERRING,
        CutoverPhase.DESTINATION_IMPORTING,
        CutoverPhase.DESTINATION_VALIDATING,
        CutoverPhase.COMMIT_INTENT_RECORDED,
        CutoverPhase.OWNERSHIP_COMMITTED,
        CutoverPhase.GATEWAY_SWITCHING,
        CutoverPhase.DESTINATION_ACTIVE,
        CutoverPhase.SOURCE_DRAINING,
        CutoverPhase.COMPLETED,
    )
)
_FAILED_PRECOMMIT_PHASES = (
    *_SUCCESS_PHASES[:11],
    CutoverPhase.ABORTING.value,
    CutoverPhase.ROLLED_BACK.value,
)


def _verify_flagship_manifest(result: FlagshipDemoResult, artifact: Path) -> None:
    manifest_path = artifact.parent / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            "flagship manifest is required; raw evaluation artifacts must be verified "
            "through their hash-validated evaluation index"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("flagship manifest is not valid JSON") from error
    expected = {
        "schema_version": "sloforge.continuum.demo-manifest/v1",
        "artifact": artifact.name,
        "sha256": sha256_file(artifact),
        "run_id": result.run_id,
        "seed": result.seed,
        "accepted_token_count": len(result.accepted_token_indices),
        "failed_transaction": result.failed_migration.transaction_id,
        "successful_transaction": result.successful_migration.transaction_id,
        "synthetic_hardware": True,
    }
    if (
        not isinstance(manifest, dict)
        or set(manifest) != set(expected)
        or any(manifest.get(key) != value for key, value in expected.items())
    ):
        raise RuntimeError("flagship manifest seal does not match the migration artifact")


def _verify_timeline(result: FlagshipDemoResult) -> None:
    if tuple(item.sequence for item in result.timeline) != tuple(range(len(result.timeline))):
        raise RuntimeError("flagship timeline sequence is not contiguous")
    for event in result.timeline:
        material = {
            "sequence": event.sequence,
            "category": event.category,
            "label": event.label,
            "transaction_id": event.transaction_id,
            "phase": event.phase,
            "session_id": event.session_id,
            "owner_epoch": event.owner_epoch,
            "token_index": event.token_index,
            "byte_count": event.byte_count,
        }
        encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
        if sha256(encoded).hexdigest() != event.evidence_digest:
            raise RuntimeError("flagship timeline evidence digest is invalid")


def _verify_flagship_evidence(
    result: FlagshipDemoResult,
    *,
    artifact: Path,
) -> dict[str, bool]:
    """Derive every declared invariant from typed transaction and event evidence."""

    _verify_flagship_manifest(result, artifact)
    _verify_timeline(result)
    validate_capsule(result.successful_migration.capsule)
    failed = result.failed_migration
    successful = result.successful_migration
    fork = result.fork
    compatibility = result.compatibility_case
    conversion = successful.live_conversion_evidence
    accepted = result.accepted_token_indices
    logical = successful.capsule.logical_state
    physical = successful.capsule.physical_state
    attention = logical.attention
    if attention is None:
        raise RuntimeError("flagship capsule lacks attention state for conversion verification")
    compared_attention_bytes = sum(
        (layer.token_range.end_exclusive - layer.token_range.start)
        * layer.kv_head_count
        * layer.head_dimension
        * 2
        * 4
        for layer in attention.layers
    )
    token_count = len(logical.token_history.input_token_ids) + len(
        logical.token_history.committed_output_token_ids
    )
    destination_pages = (
        token_count + result.destination.page_size_tokens - 1
    ) // result.destination.page_size_tokens
    non_attention_segments = sum(
        segment.logical_state_reference != "state/attention-kv" for segment in physical.segments
    )
    expected_destination_segments = (
        destination_pages * len(attention.layers) * result.destination.tensor_parallel_degree
        + non_attention_segments
    )
    conversion_hashes_valid = (
        re.fullmatch(r"[0-9a-f]{64}", conversion.source_hash) is not None
        and re.fullmatch(r"[0-9a-f]{64}", conversion.direct_hash) is not None
        and conversion.source_hash != conversion.direct_hash
    )
    conversion_valid = (
        conversion.canonical_attention_match
        and conversion_hashes_valid
        and conversion.source_segment_count == len(physical.segments)
        and conversion.destination_segment_count == expected_destination_segments
        and conversion.compared_attention_bytes == compared_attention_bytes
        and 0 < conversion.maximum_temporary_bytes <= compared_attention_bytes
        and conversion.maximum_temporary_bytes % 4 == 0
    )
    if not conversion_valid:
        raise RuntimeError("direct conversion evidence does not match capsule-derived bounds")
    if successful.phase_history != _SUCCESS_PHASES:
        raise RuntimeError("successful migration phase history is incomplete or out of order")
    if failed.phase_history != _FAILED_PRECOMMIT_PHASES:
        raise RuntimeError("failed migration did not execute the declared rollback window")
    if compatibility.recomputation_execution.phase_history != _SUCCESS_PHASES:
        raise RuntimeError("recomputation activation transaction is incomplete or out of order")
    if accepted != successful.accepted_token_indices:
        raise RuntimeError("top-level and transaction gateway ledgers disagree")
    if accepted != tuple(range(len(accepted))) or len(accepted) != len(set(accepted)):
        raise RuntimeError("gateway evidence contains a duplicate or token gap")
    main_session = successful.capsule.identity.session_id
    main_token_events = tuple(
        item
        for item in result.timeline
        if item.category == "token" and item.session_id == main_session
    )
    if tuple(item.token_index for item in main_token_events) != accepted:
        raise RuntimeError("gateway ledger is not reproduced by the event timeline")
    cutover = successful.cutover_token_index
    if any(
        item.owner_epoch
        != (
            successful.source_owner_epoch
            if item.token_index is not None and item.token_index <= cutover
            else successful.destination_owner_epoch
        )
        for item in main_token_events
    ):
        raise RuntimeError("timeline token ownership does not match the cutover boundary")
    rollback_events = [
        item.sequence
        for item in result.timeline
        if item.transaction_id == failed.transaction_id
        and item.phase == CutoverPhase.ROLLED_BACK.value
    ]
    success_proposals = [
        item.sequence
        for item in result.timeline
        if item.transaction_id == successful.transaction_id
        and item.phase == CutoverPhase.PROPOSED.value
    ]
    stale_rejections = [
        item
        for item in result.timeline
        if item.transaction_id == successful.transaction_id
        and item.label == "stale source generation rejected after ownership commit"
        and item.owner_epoch == successful.source_owner_epoch
    ]
    branches = fork.branches
    branch_structure_valid = all(
        branch.emitted_token_indices
        == tuple(range(len(accepted), len(accepted) + len(branch.emitted_token_indices)))
        and len(branch.emitted_token_indices) == len(branch.emitted_token_ids)
        and branch.emitted_token_ids[:1] == (branch.initial_next_token,)
        and branch.copy_on_write_new_chunks > 0
        and branch.copy_on_write_new_bytes > 0
        for branch in branches
    )
    recomputation = compatibility.recomputation_execution
    recomputation_valid = (
        recomputation.recomputation.destination_model_hash == compatibility.changed_model_hash
        and recomputation.recomputation.token_count == compatibility.recomputation_token_count
        and recomputation.resumed_token_ids == recomputation.recomputation.first_run_tokens
        and recomputation.imported_structurally_valid
        and recomputation.imported_continuation_valid
    )
    derived = {
        "rollback_preserved_source_epoch": (
            failed.source_epoch_before == failed.source_epoch_after == successful.source_owner_epoch
        ),
        "rollback_preserved_gateway_watermark": (
            failed.gateway_watermark_before == failed.gateway_watermark_after
            and failed.accepted_token_indices == tuple(range(failed.gateway_watermark_after + 1))
        ),
        "coordinator_restart_recovered_rollback": (
            bool(rollback_events)
            and len(success_proposals) == 1
            and max(rollback_events) < success_proposals[0]
        ),
        "cross_adapter": result.source.adapter_version != result.destination.adapter_version,
        "cross_layout": result.source.layout != result.destination.layout,
        "tensor_parallel_changed": (
            result.source.tensor_parallel_degree != result.destination.tensor_parallel_degree
        ),
        "page_size_changed": result.source.page_size_tokens != result.destination.page_size_tokens,
        "no_gateway_duplicate": len(accepted) == len(set(accepted)),
        "no_gateway_gap": accepted == tuple(range(len(accepted))),
        "stale_source_rejected": len(stale_rejections) == 1,
        "fork_sessions_distinct": (
            branches[0].session_id != branches[1].session_id
            and all(branch.session_id != fork.parent_session_id for branch in branches)
            and branch_structure_valid
        ),
        "fork_owners_distinct": (
            branches[0].owner_id != branches[1].owner_id
            and branches[0].owner_epoch != branches[1].owner_epoch
            and fork.full_copy_baseline_bytes == 2 * fork.content_addressed_checkpoint_bytes
            and fork.checkpoint_bytes_deduplicated == fork.content_addressed_checkpoint_bytes
        ),
        "unsafe_weight_reuse_rejected": (
            compatibility.source_model_hash != compatibility.changed_model_hash
            and compatibility.shapes_match
            and not compatibility.direct_reuse.safe
            and compatibility.direct_reuse.compatibility_class.value == "incompatible"
        ),
        "recomputation_plan_generated": (
            compatibility.recomputation_assisted.safe
            and compatibility.recomputation_assisted.compatibility_class.value
            == "recomputation_assisted"
        ),
        "recomputation_executed": recomputation_valid,
    }
    if not all(derived.values()):
        failed_names = sorted(name for name, passed in derived.items() if not passed)
        raise RuntimeError(f"migration evidence failed derived invariants: {failed_names}")
    if result.invariants.model_dump(mode="python") != derived:
        raise RuntimeError("claimed flagship invariants differ from independently derived evidence")
    expected_run_id = sha256(
        (
            f"continuum-flagship:{main_session}:{result.seed}:"
            f"{successful.transaction_id}:{fork.parent_manifest_id}"
        ).encode()
    ).hexdigest()
    if result.run_id != expected_run_id:
        raise RuntimeError("flagship run identifier does not bind its raw transaction evidence")
    return derived


@continuum_app.command("compatibility")
def compatibility_command(
    artifact: Annotated[Path, typer.Option("--artifact", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
) -> None:
    """Extract the evidence-backed safe and rejected model-revision decisions."""

    result = FlagshipDemoResult.model_validate_json(artifact.read_text(encoding="utf-8"))
    _verify_flagship_evidence(result, artifact=artifact)
    _write_output(output, result.compatibility_case.model_dump(mode="json"))


@continuum_app.command("fork")
def fork_command(
    artifact: Annotated[Path, typer.Option("--artifact", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
) -> None:
    """Extract the content-addressed copy-on-write branch evidence."""

    result = FlagshipDemoResult.model_validate_json(artifact.read_text(encoding="utf-8"))
    _verify_flagship_evidence(result, artifact=artifact)
    _write_output(output, result.fork.model_dump(mode="json"))


@continuum_app.command("chaos")
def chaos_command(
    scenario: Annotated[Path, typer.Option("--scenario", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    reset: Annotated[bool, typer.Option("--reset")] = False,
) -> None:
    """Execute the deterministic destination-crash-before-commit scenario."""

    document = yaml.safe_load(scenario.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema_version") != (
        "sloforge.continuum.fault-scenario/v1"
    ):
        raise typer.BadParameter("unsupported Continuum fault scenario schema")
    seed_value = document.get("seed")
    if not isinstance(seed_value, int):
        raise typer.BadParameter("fault scenario requires an integer seed")
    _prepare_output(output, reset=reset)
    result = run_flagship_demo(
        FlagshipDemoRequest(
            work_dir=output / "work",
            session_id="session-fault-001",
            tenant_id="local-continuum-fault",
            seed=seed_value,
            git_commit=git_commit(Path(__file__).resolve().parents[3]),
        )
    )
    expected = document.get("fault", {}).get("ground_truth_label")
    observed = result.failed_migration.fault.definition.ground_truth_label
    if observed != expected:
        raise RuntimeError(f"fault mismatch: expected {expected!r}, observed {observed!r}")
    artifact = output / "fault-result.json"
    write_flagship_artifact(result, artifact)
    _write_output(
        output / "fault-manifest.json",
        {
            "schema_version": "sloforge.continuum.fault-result/v1",
            "scenario": str(scenario),
            "artifact": artifact.name,
            "sha256": sha256_file(artifact),
            "ground_truth_label": observed,
            "final_phase": result.failed_migration.phase_history[-1],
            "source_epoch_unchanged": result.invariants.rollback_preserved_source_epoch,
            "gateway_watermark_unchanged": result.invariants.rollback_preserved_gateway_watermark,
        },
    )


@continuum_app.command("benchmark")
def benchmark_command(
    output: Annotated[Path, typer.Option("--output", "-o")],
    seeds: Annotated[str, typer.Option("--seeds")] = "101,202,303,404,505",
    reset: Annotated[bool, typer.Option("--reset")] = False,
) -> None:
    """Run the multi-seed CPU migration, conversion, planner, and fork matrix."""

    try:
        parsed_seeds = tuple(int(item) for item in seeds.split(",") if item)
    except ValueError as error:
        raise typer.BadParameter("seeds must be comma-separated integers") from error
    _prepare_output(output, reset=reset)
    campaign = run_evaluation_campaign(
        EvaluationRequest(
            output_dir=output,
            seeds=parsed_seeds,
            git_commit=git_commit(Path(__file__).resolve().parents[3]),
        )
    )
    typer.echo(str(output / campaign.summary_artifact.path))


@continuum_app.command("report")
def report_command(
    evaluation: Annotated[Path, typer.Option("--evaluation", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
) -> None:
    """Regenerate static reports only after every indexed raw artifact validates."""

    bundle = load_evaluation(evaluation)
    report_set = generate_reports(bundle, root=evaluation.parent)
    _write_output(output, report_set.model_dump(mode="json"))
