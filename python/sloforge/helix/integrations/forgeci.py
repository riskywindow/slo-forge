"""Build portable ForgeCI inputs from failed Helix learning transactions."""

from __future__ import annotations

import hashlib
from pathlib import Path

from sloforge.forgeci import BenchmarkMatrix
from sloforge.helix.transactions import LearningState, LearningTransactionRecord, TransactionEvent

from .models import ForgeCIRegressionArtifact, canonical_digest

_LIMITATIONS = (
    "The matrix is a deterministic reproduction request, not a measured regression result.",
    "Raw Helix evidence remains externally referenced and must be fetched and hash-verified.",
    "The failed classification describes the Helix transaction; ForgeCI determines benchmark "
    "regression classifications from new measurements.",
)


def build_forgeci_regression_artifact(
    transaction: LearningTransactionRecord,
    failure_event: TransactionEvent,
    matrix: BenchmarkMatrix,
) -> ForgeCIRegressionArtifact:
    """Seal a failed transaction and unmodified evidence references with a ForgeCI matrix."""

    if not transaction.state.terminal or transaction.state is LearningState.COMPLETED:
        raise ValueError("ForgeCI bridge requires a failed terminal Helix transaction")
    if failure_event.sequence != transaction.sequence:
        raise ValueError("failure event is not the terminal transaction event")
    if failure_event.state_after is not transaction.state:
        raise ValueError("failure event does not match the failed transaction state")
    if not transaction.artifacts:
        raise ValueError("failed transaction has no raw evidence artifacts")
    if any(not artifact.immutable for artifact in transaction.artifacts):
        raise ValueError("ForgeCI bridge requires immutable raw evidence artifacts")
    artifact_ids = {artifact.artifact_id for artifact in transaction.artifacts}
    if not set(failure_event.evidence_artifact_ids).issubset(artifact_ids):
        raise ValueError("failure event references evidence absent from the transaction")

    matrix_sha256 = canonical_digest(matrix)
    raw_evidence_set_sha256 = canonical_digest(
        [artifact.model_dump(mode="json") for artifact in transaction.artifacts]
    )
    unsealed = ForgeCIRegressionArtifact.model_construct(
        artifact_id="0" * 64,
        transaction_id=transaction.transaction_id,
        failed_state=transaction.state,
        transaction_evidence_sha256=transaction.evidence_hash,
        seed=transaction.seed,
        failure_event=failure_event,
        matrix=matrix,
        matrix_sha256=matrix_sha256,
        raw_evidence_artifacts=transaction.artifacts,
        raw_evidence_set_sha256=raw_evidence_set_sha256,
        limitations=_LIMITATIONS,
    )
    artifact_id = canonical_digest(unsealed.model_dump(mode="json", exclude={"artifact_id"}))
    return ForgeCIRegressionArtifact(
        artifact_id=artifact_id,
        transaction_id=transaction.transaction_id,
        failed_state=transaction.state,
        transaction_evidence_sha256=transaction.evidence_hash,
        seed=transaction.seed,
        failure_event=failure_event,
        matrix=matrix,
        matrix_sha256=matrix_sha256,
        raw_evidence_artifacts=transaction.artifacts,
        raw_evidence_set_sha256=raw_evidence_set_sha256,
        limitations=_LIMITATIONS,
    )


def write_forgeci_regression_artifact(artifact: ForgeCIRegressionArtifact, path: Path) -> str:
    """Write a portable canonical JSON artifact and return its file digest."""

    payload = artifact.model_dump_json(indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = ["build_forgeci_regression_artifact", "write_forgeci_regression_artifact"]
