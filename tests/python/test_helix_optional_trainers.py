from __future__ import annotations

from sloforge.helix.trainers.optional import (
    _validation_class_for_device,
    probe_optional_pytorch,
    probe_peft_lora,
)


def test_optional_trainer_probes_fail_closed_without_dependencies() -> None:
    pytorch = probe_optional_pytorch()
    peft = probe_peft_lora()
    assert pytorch.adapter == "pytorch"
    assert peft.adapter == "peft_lora"
    assert pytorch.official_api_evidence
    assert peft.official_api_evidence
    if not peft.available:
        assert peft.reason


def test_validation_class_reports_the_device_that_actually_trained() -> None:
    assert _validation_class_for_device("cpu") == "local-cpu-framework"
    assert _validation_class_for_device("mps") == "local-cpu-framework"
    assert _validation_class_for_device("cuda") == "hardware-backed"
