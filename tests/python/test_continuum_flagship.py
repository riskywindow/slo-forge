from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from sloforge.continuum.demo import (
    FlagshipDemoRequest,
    FlagshipDemoResult,
    run_flagship_demo,
    write_flagship_artifact,
)


def _request(work_dir: Path) -> FlagshipDemoRequest:
    return FlagshipDemoRequest(
        work_dir=work_dir,
        session_id="session-flagship-test",
        tenant_id="tenant-a",
        seed=317,
        initial_output_tokens=12,
        successful_delta_rounds=(2, 1),
        resumed_tokens=2,
        git_commit="7e51ea7f7338755d23f889820558a4e046d6c42e",
    )


def test_cpu_flagship_failed_cutover_recovers_then_migrates_and_forks(tmp_path: Path) -> None:
    result = run_flagship_demo(_request(tmp_path / "run"))

    assert result.failed_migration.phase_history[-1] == "ROLLED_BACK"
    assert result.failed_migration.source_lifecycle_after == "active"
    assert result.failed_migration.failure_code == "adapter_unavailable"
    assert result.failed_migration.recurrent_present
    assert result.failed_migration.sampler_present
    assert result.failed_migration.guided_decoding_present
    assert result.failed_migration.attention_segment_count > 0
    assert result.failed_migration.fault.definition.ground_truth_label == (
        "continuum.fault.destination_crash_during_validation"
    )

    successful = result.successful_migration
    assert successful.phase_history[-1] == "COMPLETED"
    assert successful.source_owner_epoch == 1
    assert successful.destination_owner_epoch == 2
    assert successful.source_runtime != successful.destination_runtime
    assert successful.source_layout != successful.destination_layout
    assert successful.capsule.logical_state.attention is not None
    assert successful.capsule.logical_state.recurrent
    assert successful.stale_source_rejected

    assert result.accepted_token_indices == tuple(range(len(result.accepted_token_indices)))
    assert all(result.invariants.model_dump(mode="python").values())
    assert result.fork.checkpoint_bytes_deduplicated > 0
    assert result.fork.shared_chunk_count > 0
    branch_a, branch_b = result.fork.branches
    assert branch_a.session_id != branch_b.session_id
    assert branch_a.owner_id != branch_b.owner_id
    assert branch_a.runtime.layout != branch_b.runtime.layout
    assert branch_a.runtime.tensor_parallel_degree != branch_b.runtime.tensor_parallel_degree
    assert branch_a.incremental_manifest_id != branch_b.incremental_manifest_id

    case = result.compatibility_case
    assert case.shapes_match
    assert not case.direct_reuse.safe
    assert case.direct_reuse.compatibility_class.value == "incompatible"
    assert {reason.code for reason in case.direct_reuse.reasons} == {
        "STATE_PRODUCING_DEPENDENCY_CHANGED"
    }
    assert case.recomputation_assisted.safe
    assert case.recomputation_assisted.compatibility_class.value == ("recomputation_assisted")
    assert all(
        "model" not in reason.message
        for reason in case.recomputation_assisted.reasons
        if reason.code == "IDENTICAL_COMPATIBILITY_DOMAIN"
    )
    assert case.recomputation_assisted.required_recomputation[0].state_components == (
        "attention.kv",
    )
    executed = case.recomputation_execution
    assert executed.phase_history[-1] == "COMPLETED"
    assert executed.destination_owner_epoch == executed.source_owner_epoch + 1
    assert executed.resumed_token_ids == executed.recomputation.first_run_tokens
    assert executed.imported_structurally_valid
    assert executed.imported_continuation_valid

    fault_events = [event for event in result.timeline if event.category == "fault"]
    assert len(fault_events) == 1
    assert fault_events[0].phase == "DESTINATION_VALIDATING"
    assert tuple(event.sequence for event in result.timeline) == tuple(range(len(result.timeline)))


def test_flagship_is_seed_deterministic_and_artifact_round_trips(tmp_path: Path) -> None:
    first = run_flagship_demo(_request(tmp_path / "first"))
    second = run_flagship_demo(_request(tmp_path / "second"))

    assert first == second
    output = tmp_path / "artifacts" / "flagship.json"
    write_flagship_artifact(first, output)
    restored = FlagshipDemoResult.model_validate_json(output.read_text(), strict=True)
    assert restored == first
    assert "attention_keys" not in output.read_text()

    tampered = first.model_dump(mode="json")
    tampered["compatibility_case"]["recomputation_execution"]["resumed_token_ids"][0] += 1
    with pytest.raises(ValidationError, match=r"evidence digest|diverged"):
        FlagshipDemoResult.model_validate_json(json.dumps(tampered), strict=True)
