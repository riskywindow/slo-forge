from __future__ import annotations

import pytest
from pydantic import ValidationError

from sloforge.continuum.faults import (
    FAULT_CATALOG,
    FaultActivation,
    FaultKind,
    fault_definition,
)


def test_fault_catalog_covers_the_declared_deterministic_matrix() -> None:
    assert {definition.kind for definition in FAULT_CATALOG} == set(FaultKind)
    assert len({definition.ground_truth_label for definition in FAULT_CATALOG}) == len(FaultKind)
    assert all(definition.transaction_phase for definition in FAULT_CATALOG)
    assert all(definition.expected_protocol_response for definition in FAULT_CATALOG)
    assert not any(definition.host_wide for definition in FAULT_CATALOG)


def test_fault_activation_is_json_ready_and_interval_checked() -> None:
    definition = fault_definition(FaultKind.DESTINATION_CRASH_DURING_VALIDATION)
    activation = FaultActivation(
        definition=definition,
        transaction_id="transaction-1",
        activation_sequence=10,
        clear_sequence=12,
        observed_protocol_response="pre-commit rollback retained the source owner",
        injected=True,
    )

    assert (
        FaultActivation.model_validate_json(activation.model_dump_json(), strict=True) == activation
    )
    assert activation.definition.affected_component.value == "destination_runtime"
    assert activation.definition.transaction_phase == "DESTINATION_VALIDATING"

    with pytest.raises(ValidationError, match="clear sequence"):
        FaultActivation(
            definition=definition,
            transaction_id="transaction-1",
            activation_sequence=10,
            clear_sequence=9,
            observed_protocol_response="invalid interval",
            injected=True,
        )
