from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from sloforge.autopsy import AutopsyRun, SourceClock
from sloforge.forgeci import (
    BenchmarkMatrix,
    BenchmarkSpec,
    CommandSpec,
    HardwareRequirement,
    MatrixCase,
    MetricDirection,
    MetricSpec,
    load_matrix,
    write_matrix,
)
from sloforge.helix.integrations import (
    BranchOperationEvidence,
    BranchOperationKind,
    BranchTraceExport,
    ForgeCIRegressionArtifact,
    build_forgeci_regression_artifact,
    export_branch_workload_trace,
    write_forgeci_regression_artifact,
)
from sloforge.helix.ir import BranchWorkloadTrace, canonical_hash, load_learning_transaction
from sloforge.helix.transactions import (
    ArtifactReference,
    LearningState,
    LearningTransactionStore,
)

ROOT = Path(__file__).parents[2]
FIXTURE = ROOT / "tests/fixtures/helix/learning-transaction-v1.json"


def _branch_export() -> BranchTraceExport:
    transaction = load_learning_transaction(FIXTURE)
    return export_branch_workload_trace(
        transaction.branch_group,
        raw_branch_group_uri="fixture://helix/learning-transaction-v1#branch-group",
        scheduled_offsets_ms=(0.0, 12.5),
        seed=20260803,
        topology_fingerprint=hashlib.sha256(b"test-topology").hexdigest(),
        physical_plan_hash=hashlib.sha256(b"test-physical-plan").hexdigest(),
    )


def _matrix(seed: int) -> BenchmarkMatrix:
    return BenchmarkMatrix(
        matrix_id="helix-failed-transaction",
        cases=(
            MatrixCase(
                case_id="reproduce-helix-failure",
                repository="https://example.invalid/sloforge.git",
                revision="fixture-revision",
                hardware=HardwareRequirement(architecture="cpu"),
                benchmark=BenchmarkSpec(
                    command=CommandSpec(executable="python", arguments=("reproduce_failure.py",)),
                    metrics=(
                        MetricSpec(
                            name="failure_rate",
                            unit="ratio",
                            direction=MetricDirection.LOWER_IS_BETTER,
                        ),
                    ),
                    warmup_trials=0,
                    repetitions=3,
                    maximum_repetitions=3,
                    bootstrap_rounds=200,
                    seed=seed,
                ),
            ),
        ),
    )


def _failed_transaction(tmp_path: Path) -> tuple[LearningTransactionStore, Path]:
    raw_path = tmp_path / "raw-failure.json"
    raw_path.write_bytes(b'{"failure":"privacy authorization absent"}\n')
    raw_digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    store = LearningTransactionStore(tmp_path / "transactions.sqlite")
    store.create(
        transaction_id="helix-tx-failed",
        deployment="coding-agent-prod",
        champion_policy_epoch_id="champion-17",
        trigger_hash=hashlib.sha256(b"failure trigger").hexdigest(),
        seed=47,
        observed_at_ms=1,
    )
    artifact = ArtifactReference(
        artifact_id="failure-evidence-1",
        artifact_kind="production_failure",
        sha256=raw_digest,
        uri=raw_path.as_uri(),
    )
    store.add_artifact("helix-tx-failed", artifact)
    store.transition(
        "helix-tx-failed",
        target=LearningState.EXPERIENCE_REJECTED,
        reason="privacy authorization absent",
        observed_at_ms=2,
        evidence_artifact_ids=(artifact.artifact_id,),
    )
    return store, raw_path


def test_branch_trace_exports_real_capsule_evidence_deterministically() -> None:
    transaction = load_learning_transaction(FIXTURE)
    exported = _branch_export()

    assert exported == _branch_export()
    assert isinstance(exported.workload_trace, BranchWorkloadTrace)
    assert exported.source_branch_group_sha256 == canonical_hash(transaction.branch_group)
    assert [request.ordinal for request in exported.workload_trace.requests] == [0, 1]
    assert [request.scheduled_offset_ms for request in exported.workload_trace.requests] == [
        0.0,
        12.5,
    ]
    assert {operation.kind for operation in exported.operations} == set(BranchOperationKind)

    capsule = transaction.branch_group.branch_point.environment_state
    transfers = [
        operation
        for operation in exported.operations
        if operation.kind is BranchOperationKind.TRANSFER
    ]
    assert transfers
    assert all(operation.byte_count == capsule.payload_byte_length for operation in transfers)
    assert all("no physical transfer measured" in operation.detail for operation in transfers)
    divergence = next(
        operation
        for operation in exported.operations
        if operation.kind is BranchOperationKind.DIVERGENCE
    )
    assert "canonical trajectory capsule" in divergence.detail

    run = AutopsyRun.model_validate_json(exported.autopsy_run.model_dump_json())
    assert run == exported.autopsy_run
    assert all(event.source_clock is SourceClock.SYNTHETIC for event in run.events)
    assert all(event.duration_ns == 0 for event in run.events)
    assert "not latency measurements" in run.warnings[0]
    assert tuple(event.event_id for event in run.events) == tuple(
        operation.operation_id for operation in exported.operations
    )


def test_branch_trace_rejects_invalid_schedule_and_seal_tampering() -> None:
    transaction = load_learning_transaction(FIXTURE)
    with pytest.raises(ValueError, match="one scheduled offset"):
        export_branch_workload_trace(
            transaction.branch_group,
            raw_branch_group_uri="fixture://helix/branch-group",
            scheduled_offsets_ms=(0.0,),
            seed=9,
            topology_fingerprint=hashlib.sha256(b"topology").hexdigest(),
            physical_plan_hash=hashlib.sha256(b"plan").hexdigest(),
        )
    with pytest.raises(ValueError, match="finite and non-negative"):
        export_branch_workload_trace(
            transaction.branch_group,
            raw_branch_group_uri="fixture://helix/branch-group",
            scheduled_offsets_ms=(0.0, float("nan")),
            seed=9,
            topology_fingerprint=hashlib.sha256(b"topology").hexdigest(),
            physical_plan_hash=hashlib.sha256(b"plan").hexdigest(),
        )

    exported = _branch_export()
    operation_payload = exported.operations[0].model_dump(mode="json")
    operation_payload["checksum_sha256"] = hashlib.sha256(b"tampered").hexdigest()
    with pytest.raises(ValidationError, match="identifier is invalid"):
        BranchOperationEvidence.model_validate_json(json.dumps(operation_payload))

    export_payload = exported.model_dump(mode="json")
    export_payload["source_branch_group_sha256"] = hashlib.sha256(b"tampered").hexdigest()
    with pytest.raises(ValidationError, match="export identifier is invalid"):
        BranchTraceExport.model_validate_json(json.dumps(export_payload))
    export_payload = exported.model_dump(mode="json")
    export_payload["unexpected"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        BranchTraceExport.model_validate_json(json.dumps(export_payload))


def test_failed_transaction_builds_portable_forgeci_artifact_without_raw_mutation(
    tmp_path: Path,
) -> None:
    store, raw_path = _failed_transaction(tmp_path)
    try:
        transaction = store.transaction("helix-tx-failed")
        event = store.events(transaction.transaction_id)[-1]
        before = raw_path.read_bytes()
        artifact = build_forgeci_regression_artifact(transaction, event, _matrix(transaction.seed))
        output_path = tmp_path / "portable" / "helix-forgeci-regression.json"
        file_digest = write_forgeci_regression_artifact(artifact, output_path)

        assert raw_path.read_bytes() == before
        assert file_digest == hashlib.sha256(output_path.read_bytes()).hexdigest()
        assert artifact.transaction_evidence_sha256 == transaction.evidence_hash
        assert artifact.raw_evidence_artifacts == transaction.artifacts
        assert artifact.raw_evidence_unchanged is True
        assert artifact.seed == transaction.seed
        assert ForgeCIRegressionArtifact.model_validate_json(output_path.read_text()) == artifact

        matrix_path = tmp_path / "portable" / "matrix.json"
        write_matrix(artifact.matrix, matrix_path)
        assert load_matrix(matrix_path) == artifact.matrix
    finally:
        store.close()


def test_forgeci_bridge_rejects_nonterminal_mismatched_and_tampered_inputs(
    tmp_path: Path,
) -> None:
    store, _ = _failed_transaction(tmp_path)
    try:
        transaction = store.transaction("helix-tx-failed")
        event = store.events(transaction.transaction_id)[-1]
        with pytest.raises(ValueError, match="terminal transaction event"):
            build_forgeci_regression_artifact(
                transaction,
                event.model_copy(update={"sequence": event.sequence + 1}),
                _matrix(transaction.seed),
            )

        artifact = build_forgeci_regression_artifact(transaction, event, _matrix(transaction.seed))
        payload = artifact.model_dump(mode="json")
        payload["matrix_sha256"] = hashlib.sha256(b"tampered").hexdigest()
        with pytest.raises(ValidationError, match="matrix digest is invalid"):
            ForgeCIRegressionArtifact.model_validate_json(json.dumps(payload))
    finally:
        store.close()

    with LearningTransactionStore(tmp_path / "pending.sqlite") as pending:
        transaction = pending.create(
            transaction_id="helix-tx-pending",
            deployment="test",
            champion_policy_epoch_id="champion-1",
            trigger_hash=hashlib.sha256(b"pending trigger").hexdigest(),
            seed=5,
            observed_at_ms=1,
        )
        event = pending.events(transaction.transaction_id)[-1]
        with pytest.raises(ValueError, match="failed terminal"):
            build_forgeci_regression_artifact(transaction, event, _matrix(transaction.seed))
