"""Artifact-derived sharing analysis for one characterized Helix branch group."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def _events_with_operation(
    events: Iterable[Mapping[str, Any]], operation: str
) -> tuple[Mapping[str, Any], ...]:
    return tuple(event for event in events if event.get("operation_type") == operation)


def analyze_branch_state_sharing(
    branch_events: Iterable[Mapping[str, Any]],
    state_events: Iterable[Mapping[str, Any]],
) -> dict[str, object]:
    """Compute sharing without conflating CAS reuse and live workspace copies.

    Values come from the exact checkpoint chunk references and environment
    capsule/workspace sizes captured by the harness.  No page-level physical COW
    is inferred from content-addressed deduplication.
    """

    branch = tuple(branch_events)
    state = tuple(state_events)
    forks = _events_with_operation(state, "STATE_FORK")
    environment_forks = _events_with_operation(branch, "ENVIRONMENT_FORK")
    environment_checkpoints = _events_with_operation(state, "STATE_SNAPSHOT")
    environment_checkpoints = tuple(
        event for event in environment_checkpoints if event.get("state_segment") == "filesystem"
    )
    cow = _events_with_operation(state, "STATE_COW")

    model_logical = sum(int(event.get("logical_bytes", 0)) for event in forks)
    model_shared = sum(int(event.get("shared_logical_bytes", 0)) for event in forks)
    model_incremental = sum(int(event.get("physical_bytes", 0)) for event in forks)
    naive_model = sum(int(event.get("naive_independent_bytes", 0)) for event in forks)
    model_physical = (
        int(forks[0].get("source_physical_bytes", 0)) + model_incremental if forks else 0
    )

    environment_base = max(
        (int(event.get("logical_bytes", 0)) for event in environment_forks), default=0
    )
    live_workspace = sum(int(event.get("physical_bytes", 0)) for event in environment_forks)
    environment_private = sum(int(event.get("dirty_bytes", 0)) for event in environment_checkpoints)
    environment_checkpoint_incremental = sum(
        int(event.get("physical_bytes", 0)) for event in environment_checkpoints
    )
    branch_count = len(forks) or len(environment_forks)
    naive_environment = environment_base * branch_count

    # The reference checkpoint uses uncompressed plaintext CAS chunks.  Unique
    # content bytes are therefore exactly the source chunks plus newly allocated
    # child chunks; semantic-private components can still have identical content.
    unique_model_logical = model_physical
    return {
        "measurement_source": "SYNTHETIC",
        "workload_evidence_class": "SYNTHETIC",
        "timing_measurement_class": "HARDWARE_BACKED_REAL",
        "branch_count": branch_count,
        "model_state": {
            "logical_branch_bytes": model_logical,
            "shared_logical_bytes": model_shared,
            "incremental_cas_physical_bytes": model_incremental,
            "source_cas_physical_bytes": int(forks[0].get("source_physical_bytes", 0))
            if forks
            else 0,
            "physical_allocated_bytes": model_physical,
            "naive_independent_bytes": naive_model,
            "logical_unique_content_bytes": unique_model_logical,
            "sharing_efficiency": (1.0 - model_physical / naive_model if naive_model else 0.0),
            "physical_amplification": (
                model_physical / unique_model_logical if unique_model_logical else 0.0
            ),
        },
        "environment_state": {
            "base_logical_bytes": environment_base,
            "live_workspace_physical_bytes": live_workspace,
            "naive_live_workspace_bytes": naive_environment,
            "checkpoint_private_dirty_bytes": environment_private,
            "checkpoint_incremental_cas_bytes": environment_checkpoint_incremental,
            "workspace_implementation": "eager_restore_then_atomic_replace",
            "checkpoint_implementation": "content_addressed_incremental_capsule",
        },
        "copy_on_write": {
            "events": len(cow),
            "bytes_written": sum(int(event.get("bytes", 0)) for event in cow),
            "old_bytes_replaced": sum(int(event.get("old_bytes", 0)) for event in cow),
            "page_faults_observed": 0,
            "page_fault_measurement_available": False,
        },
        "limitations": (
            "The reference environment backend eagerly materializes branch workspaces; "
            "filesystem page sharing is not claimed.",
            "CAS byte counts describe application-level content deduplication, not filesystem "
            "block allocation.",
        ),
    }
