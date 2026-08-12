from __future__ import annotations

import pytest

from sloforge.helix.characterization.gpu_reclamation import (
    ExperimentPhase,
    ExperimentPhaseMarker,
)
from sloforge.helix.characterization.gpu_reclamation_trace import (
    Experiment004TraceIdentity,
    phase_markers_to_trace_events,
)
from sloforge.helix.characterization.trace import (
    BranchWorkloadEventV1,
    StateOperationEventV1,
    verify_event,
)


def _identity() -> Experiment004TraceIdentity:
    return Experiment004TraceIdentity(
        trace_id="exp004-trace",
        session_id="gpu1-rollout",
        branch_group_id="group-1",
        logical_state_id="checkpoint-1",
        tenant_id="helix",
        security_domain="helix",
        device="GPU-fixture",
        host="fixture-host",
        process_id=42,
    )


def test_every_experiment_phase_projects_to_both_trace_v1_streams() -> None:
    markers = tuple(
        ExperimentPhaseMarker(
            phase=phase,
            monotonic_timestamp_ns=100 + index,
            logical_bytes=10,
            physical_bytes=16,
        )
        for index, phase in enumerate(ExperimentPhase)
    )
    events = phase_markers_to_trace_events(markers, identity=_identity())
    assert len(events) == 2 * len(ExperimentPhase)
    assert sum(isinstance(item, BranchWorkloadEventV1) for item in events) == len(markers)
    assert sum(isinstance(item, StateOperationEventV1) for item in events) == len(markers)
    assert {item.attributes["experiment_phase"] for item in events} == {
        phase.value for phase in ExperimentPhase
    }
    assert [item.event_sequence for item in events] == list(range(len(events)))
    for event in events:
        verify_event(event)


def test_transfer_phases_preserve_direction_and_bytes() -> None:
    markers = (
        ExperimentPhaseMarker(
            phase=ExperimentPhase.D2H_BEGIN,
            monotonic_timestamp_ns=10,
            logical_bytes=8,
            physical_bytes=16,
        ),
        ExperimentPhaseMarker(
            phase=ExperimentPhase.H2D_END,
            monotonic_timestamp_ns=20,
            logical_bytes=8,
            physical_bytes=16,
        ),
    )
    events = phase_markers_to_trace_events(markers, identity=_identity())
    branches = [item for item in events if isinstance(item, BranchWorkloadEventV1)]
    assert [item.transferred_bytes for item in branches] == [16, 16]
    assert branches[0].source_location.value == "gpu_hbm"
    assert branches[1].destination_location.value == "gpu_hbm"


def test_phase_projection_rejects_nonmonotonic_markers() -> None:
    markers = (
        ExperimentPhaseMarker(phase=ExperimentPhase.STATE_CAPTURE_BEGIN, monotonic_timestamp_ns=20),
        ExperimentPhaseMarker(phase=ExperimentPhase.STATE_CAPTURE_END, monotonic_timestamp_ns=10),
    )
    with pytest.raises(ValueError, match="not monotonic"):
        phase_markers_to_trace_events(markers, identity=_identity())
